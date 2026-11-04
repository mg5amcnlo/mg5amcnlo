# SPDX-License-Identifier: 0BSD
"""Component matching of the numerical process-group analysis."""

from __future__ import annotations

import math
from fractions import Fraction
from types import SimpleNamespace

import pytest

from pyamplicol.api.process_groups import group_processes
from pyamplicol.api.results import ResolvedEvaluation
from pyamplicol.generation.process_grouping import LegMapping
from pyamplicol.generation.process_grouping_numerical import (
    PARAMETER_VARIATION,
    accepted_factor,
    color_roles,
    common_factor,
    flow_key,
    member_point,
    on_shell_point,
    parameter_variation,
    resolved_components,
)

_COLORS = {1: 3, -1: -3, 2: 3, -2: -3, 21: 8, 22: 1}
_IDENTITY = tuple(range(4))


def _resolved(
    helicities: dict[str, tuple[int, ...]],
    flows: tuple[str, ...],
    values: dict[tuple[str, str], complex],
    accuracy: str = "lc",
) -> ResolvedEvaluation:
    return ResolvedEvaluation(
        values=(
            tuple(
                tuple(values.get((helicity, flow), 0j) for flow in flows)
                for helicity in helicities
            ),
        ),
        helicity_ids=tuple(helicities),
        color_ids=flows,
        color_accuracy=accuracy,  # type: ignore[arg-type]
    )


def test_color_roles_cross_initial_fundamentals() -> None:
    # u g > u g: the incoming quark is an outgoing antifundamental.
    assert color_roles((2, 21, 2, 21), 2, _COLORS) == (-3, 8, 3, 8)
    assert color_roles((-2, 21, -2, 22), 2, _COLORS) == (3, 8, -3, 1)


def test_flow_keys_compare_lines_as_sets_and_traces_up_to_rotation() -> None:
    roles = (3, -3, 3, -3)
    first = flow_key((0, 1, 2, 3), roles, _IDENTITY, reverse=False)
    assert first == flow_key((2, 3, 0, 1), roles, _IDENTITY, reverse=False)
    assert first != flow_key((0, 3, 2, 1), roles, _IDENTITY, reverse=False)

    gluons = (8, 8, 8, 8)
    trace = flow_key((0, 1, 2, 3), gluons, _IDENTITY, reverse=False)
    assert trace == flow_key((2, 3, 0, 1), gluons, _IDENTITY, reverse=False)
    # The reflected trace is a different colour flow ...
    reflected = flow_key((0, 3, 2, 1), gluons, _IDENTITY, reverse=False)
    assert trace != reflected
    # ... which charge conjugation reaches.
    assert trace == flow_key((0, 3, 2, 1), gluons, _IDENTITY, reverse=True)
    # A relabelling moves the trace onto the target legs.
    assert flow_key((0, 1, 2, 3), gluons, (1, 0, 2, 3), reverse=False) == (
        "trace",
        (0, 2, 3, 1),
    )


def test_conjugated_lines_reverse_into_the_representative_orientation() -> None:
    # Member u~ g > u~ g, representative u g > u g, identity legs.
    member_roles = color_roles((-2, 21, -2, 21), 2, _COLORS)
    representative_roles = color_roles((2, 21, 2, 21), 2, _COLORS)
    member = flow_key((0, 1, 3, 2), member_roles, _IDENTITY, reverse=True)
    assert member == flow_key(
        (2, 3, 1, 0), representative_roles, _IDENTITY, reverse=False
    )
    # Without conjugation the line runs against the representative's.
    assert member != flow_key((0, 1, 3, 2), member_roles, _IDENTITY, reverse=False)
    assert flow_key((0, 1, 3, 2), member_roles, _IDENTITY, reverse=False) not in {
        flow_key(word, representative_roles, _IDENTITY, reverse=False)
        for word in ((2, 3, 1, 0), (2, 1, 3, 0))
    }


def test_undecodable_words_are_kept_verbatim() -> None:
    assert flow_key((0, 1), (-3, 3), (1, 0), reverse=False) == ("word", (1, 0))
    assert flow_key((0, 1), (6, -6), (0, 1), reverse=True) == ("word", (1, 0))


def _gq_process() -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
    # u g > u g g
    helicities = {
        "h:a": (-1, 1, -1, 1, 1),
        "h:b": (1, 1, 1, -1, 1),
        "h:c": (1, -1, 1, 1, -1),
    }
    flows = {"flow:x": (2, 1, 3, 4, 0), "flow:y": (2, 3, 4, 1, 0)}
    return helicities, flows


def test_resolved_components_permute_helicities_and_relabel_flows() -> None:
    # Representative u g > u g g, member u g > g u g: representative leg r is
    # member leg perm[r]. The final legs are cycled, so perm is not its own
    # inverse and the direction of the convention matters.
    permutation = (0, 1, 3, 4, 2)

    def to_member(legs: tuple[int, ...]) -> tuple[int, ...]:
        member = [0] * len(legs)
        for representative_index, value in enumerate(legs):
            member[permutation[representative_index]] = value
        return tuple(member)

    helicities, flows = _gq_process()
    pdgs = (2, 21, 2, 21, 21)
    values = {
        ("h:a", "flow:x"): 2.0,
        ("h:a", "flow:y"): 3.0,
        ("h:b", "flow:y"): 7.0,
        ("h:c", "flow:x"): 5.0,
    }
    representative = resolved_components(
        _resolved(helicities, tuple(flows), values),
        0,
        helicities=helicities,
        words=flows,
        roles=color_roles(pdgs, 2, _COLORS),
        mapping=None,
    )

    member_helicities = {key: to_member(value) for key, value in helicities.items()}
    member_flows = {
        key: tuple(permutation[i] for i in word) for key, word in flows.items()
    }
    assert to_member(pdgs) == (2, 21, 21, 2, 21)
    member_resolved = _resolved(
        member_helicities,
        tuple(member_flows),
        {key: -1.5 * value for key, value in values.items()},
    )

    def member(mapping: LegMapping) -> dict:
        return resolved_components(
            member_resolved,
            0,
            helicities=member_helicities,
            words=member_flows,
            roles=color_roles(to_member(pdgs), 2, _COLORS),
            mapping=mapping,
        )

    related = member(LegMapping(permutation, False, "flavor-profile"))
    assert related.keys() == representative.keys()
    assert common_factor([related], [representative]) == pytest.approx(-1.5)
    assert accepted_factor(
        common_factor([related], [representative]), only_unit_multiplier=False
    ) == Fraction(-3, 2)

    # The inverse permutation, the opposite convention, is rejected.
    inverse = LegMapping((0, 1, 4, 2, 3), False, "signed-pdg")
    assert common_factor([member(inverse)], [representative]) is None


def test_common_factor_requires_one_ratio_and_shared_zeros() -> None:
    key = ((1, 1), "c")
    other = ((1, -1), "c")
    third = ((-1, -1), "c")
    representative = [{key: 4.0 + 0j, other: 1.0 + 0j, third: 0j}] * 2
    assert common_factor(
        [{key: 64.0 + 0j, other: 16.0 + 0j, third: 0j}] * 2, representative
    ) == pytest.approx(16.0)
    # Components absent on one side are zero there.
    assert common_factor([{key: 64.0 + 0j, other: 16.0 + 0j}] * 2, representative)
    # Numerical noise far below the largest component is zero.
    noisy = {key: 64.0 + 0j, other: 16.0 + 0j, third: 1e-15 + 0j}
    assert common_factor([noisy] * 2, representative) == pytest.approx(16.0)
    # A zero must be zero on both sides.
    assert (
        common_factor([{key: 64.0 + 0j, other: 0j, third: 0j}] * 2, representative)
        is None
    )
    assert (
        common_factor(
            [{key: 64.0 + 0j, other: 16.0 + 0j, third: 1e-3 + 0j}] * 2,
            representative,
        )
        is None
    )
    # Components disagree, or the ratio changes between points.
    assert (
        common_factor([{key: 64.0 + 0j, other: 17.0 + 0j}] * 2, representative) is None
    )
    assert (
        common_factor(
            [
                {key: 64.0 + 0j, other: 16.0 + 0j},
                {key: 32.0 + 0j, other: 8.0 + 0j},
            ],
            representative,
        )
        is None
    )
    # A vanishing point decides nothing.
    assert common_factor([{key: 0j}], [{key: 1.0 + 0j}]) is None


def test_accepted_factor_is_rational_real_and_optionally_unit() -> None:
    assert accepted_factor(16.0 + 0j, only_unit_multiplier=False) == 16
    assert accepted_factor(16.0 + 0j, only_unit_multiplier=True) is None
    assert accepted_factor(1.0 + 1e-14j, only_unit_multiplier=True) == 1
    assert accepted_factor(1.0 + 1e-3j, only_unit_multiplier=False) is None
    assert accepted_factor(math.pi + 0j, only_unit_multiplier=False) is None
    assert accepted_factor(None, only_unit_multiplier=False) is None


def _parameter(name: str, kind: str, value: float, *, mutable: bool = True):
    return SimpleNamespace(
        name=name,
        kind=kind,
        default_real=value,
        default_imaginary=0.0,
        mutable=mutable,
    )


def test_parameter_variation_is_per_name_deterministic_and_bounded() -> None:
    parameters = [
        _parameter("aEWM1", "external", 132.5),
        _parameter("normalization.alpha_s", "normalization", 0.118),
        _parameter("MW", "derived", 80.4, mutable=False),
        _parameter("particle.6.mass", "mass", 173.0),
        _parameter("particle.6.width", "width", 1.5),
        _parameter("coupling.1.component_0", "coupling", 0.5),
        # UFO masses and widths are external inputs, but stay fixed.
        _parameter("MZ", "external", 91.188),
        _parameter("WZ", "external", 2.44),
    ]
    first = parameter_variation(parameters, point=0, fixed={"MZ", "WZ"})
    assert set(first) == {"aEWM1", "normalization.alpha_s"}
    assert first == parameter_variation(parameters[:2], point=0)
    assert first != parameter_variation(parameters, point=1, fixed={"MZ", "WZ"})
    for parameter in parameters[:2]:
        ratio = first[parameter.name] / parameter.default_real
        assert abs(ratio - 1.0) <= PARAMETER_VARIATION
        assert ratio != 1.0


def _mass_squared(momentum) -> float:
    energy, x, y, z = momentum
    return energy * energy - x * x - y * y - z * z


@pytest.mark.parametrize(
    ("initial", "final"),
    (
        ((0.0, 0.0), (0.0, 0.0, 0.0)),
        ((80.0, 173.0), (91.0, 0.0)),
        ((125.0,), (4.7, 4.7)),
        ((0.0, 0.0), (125.0,)),
    ),
)
def test_on_shell_points_conserve_momentum_with_massive_legs(initial, final) -> None:
    point = on_shell_point(initial, final, seed=7)
    assert point is not None
    count = len(initial)
    for leg, mass in zip(point, (*initial, *final), strict=True):
        assert _mass_squared(leg) == pytest.approx(mass * mass, abs=1e-9 * leg[0] ** 2)
    for component in range(4):
        incoming = sum(leg[component] for leg in point[:count])
        outgoing = sum(leg[component] for leg in point[count:])
        assert incoming == pytest.approx(outgoing, abs=1e-8)
    assert point == on_shell_point(initial, final, seed=7)


def test_on_shell_point_reports_closed_phase_space() -> None:
    assert on_shell_point((4.7,), (125.0, 0.0), seed=1) is None
    assert on_shell_point((100.0, 100.0), (150.0,), seed=1) is None


def test_member_point_follows_the_relation_convention() -> None:
    # A cycle of the final legs, which is not its own inverse.
    permutation = (0, 1, 3, 4, 2)
    representative = tuple((float(index), 0.0, 0.0, 0.0) for index in range(5))
    member = member_point(representative, permutation)
    assert [leg[0] for leg in member] == [0.0, 1.0, 4.0, 2.0, 3.0]
    for representative_index, member_index in enumerate(permutation):
        assert representative[representative_index] == member[member_index]


def test_group_processes_validates_its_arguments(tmp_path) -> None:
    with pytest.raises(ValueError, match="analysis"):
        group_processes(tmp_path, analysis="exact")
    with pytest.raises(TypeError, match="only_unit_multiplier"):
        group_processes(tmp_path, only_unit_multiplier=1)  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        group_processes(tmp_path / "missing")
