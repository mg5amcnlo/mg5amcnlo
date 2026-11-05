# SPDX-License-Identifier: 0BSD
"""Colour coefficients must not impose a binary64 floor on exact evaluation."""

from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from pyamplicol.api.errors import ArtifactError
from pyamplicol.color import (
    ColorGroupDescriptor,
    build_color_contraction_plan,
    build_color_plan,
)
from pyamplicol.color.contraction_factors import exact_color_contraction_factor
from pyamplicol.generation.artifact_writer import _color_contraction
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.runtime._color_topology_exact import (
    _parse_expanded_entry,
    _parse_repeated_contraction,
)
from pyamplicol.runtime._color_weight_exact import exact_color_weight
from pyamplicol.runtime.symbolica_exact import _validated_color_contraction_entries


@pytest.mark.parametrize("accuracy", ("nlc", "full"))
@pytest.mark.parametrize("components", (1, 2))
def test_generated_colour_weights_retain_their_rational_source(accuracy, components):
    plan = build_color_plan(
        build_process_ir("g g > g g", color_accuracy=accuracy),
        color_accuracy=accuracy,
    )
    sectors = plan.sectors[:2]
    groups = tuple(
        ColorGroupDescriptor(
            group_id=2 * component + index,
            helicity_key=(component,),
            sector_id=sector.id,
            word=tuple(sector.word_labels or sector.color_words[0]),
            helicity_weight=2.0,
        )
        for component in range(components)
        for index, sector in enumerate(sectors)
    )
    contraction = build_color_contraction_plan(plan, groups)
    assert contraction is not None
    by_group = {
        group.group_id: sectors[index % 2] for index, group in enumerate(groups)
    }
    for entry in contraction.iter_logical_entries():
        expected = 2 * exact_color_contraction_factor(
            plan,
            by_group[entry.left_group_id],
            by_group[entry.right_group_id],
            accuracy=accuracy,
        )
        assert entry.exact_weight == (expected, Fraction(0))
        assert entry.weight_re == float(expected)
    compact = _color_contraction(contraction.to_json_dict())
    entries = (
        compact["entries"] if components == 1 else compact["repeated_block"]["entries"]
    )
    assert entries and all("exact_weight" in entry for entry in entries)


def test_expanded_and_repeated_exact_readers_do_not_upcast_f64_colour_weights():
    weight = {
        "weight": [1 / 3, -2 / 7],
        "exact_weight": ["1", "3", "-2", "7"],
        "symmetry_factor": 1.0,
    }
    expanded = {
        "group_count": 2,
        "entries": [dict(weight, left_group_id=i, right_group_id=i) for i in range(2)],
    }
    repeated = {
        "group_count": 2,
        "entries": [],
        "repeated_block": {
            "component_count": 2,
            "component_group_ids": [0, 1],
            "entries": [dict(weight, left_group_index=0, right_group_index=0)],
        },
    }
    with localcontext() as context:
        context.prec = 80
        expected = Decimal(1) / 3, -Decimal(2) / 7
        for contraction in (expanded, repeated):
            entries = tuple(
                _validated_color_contraction_entries(contraction, {0: None, 1: None})
            )
            assert [entry.weight for entry in entries] == [expected, expected]
        assert _parse_expanded_entry(expanded["entries"][0], 2).weight == expected
        assert (
            _parse_repeated_contraction(repeated["repeated_block"], 2).entries[0].weight
            == expected
        )


@pytest.mark.parametrize(
    "raw", (["1", "0", "0", "1"], ["1.5", "3", "0", "1"], [1, 3, 0, 1], ["1", "3"])
)
def test_exact_colour_weight_rejects_invalid_rational_payload(raw):
    with pytest.raises(ArtifactError, match="exact colour weight"):
        exact_color_weight({"exact_weight": raw})
