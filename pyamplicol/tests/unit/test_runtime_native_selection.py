# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from decimal import Decimal
from fractions import Fraction

import pytest

from pyamplicol.api.errors import ArtifactError
from pyamplicol.api.results import ResolvedEvaluation
from pyamplicol.runtime._native_selection import (
    charge_conjugate_color_id,
    native_physics_axes,
    native_process_group_relation,
    native_process_selection,
    remap_reduction,
    representative_vector_to_public,
)
from pyamplicol.runtime.backend import _ProcessGroupExactExecutor


class _NativeRuntime:
    def __init__(
        self,
        payload: dict[str, object],
        *,
        physics: dict[str, object] | None = None,
    ) -> None:
        self._payload = payload
        self._physics = physics

    def _exact_runtime_state_json(self) -> str:
        return json.dumps(self._payload)

    def physics_json(self) -> str:
        return json.dumps(self._physics)


def _state(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "model_parameter_values": [],
        "normalization_factor": 1.0,
        "representative_process_id": "representative",
        "representative_process_key": "representative-key",
        "external_permutation": [1, 0, 3, 2],
    }
    payload.update(updates)
    return payload


_PROCESSES = (
    {
        "id": "representative",
        "external_pdgs": [2, -2, 23, 21],
        "aliases": [],
    },
)


def test_native_selection_is_authoritative_for_inferred_public_expression() -> None:
    selection = native_process_selection(_NativeRuntime(_state()), _PROCESSES)

    assert selection.process is _PROCESSES[0]
    assert selection.representative_process_id == "representative"
    assert selection.representative_process_key == "representative-key"
    assert selection.external_permutation == (1, 0, 3, 2)


@pytest.mark.parametrize(
    "updates",
    [
        {"representative_process_id": "missing"},
        {"representative_process_key": ""},
        {"external_permutation": [0, 1, 2]},
        {"external_permutation": [0, 1, 2, 2]},
        {"external_permutation": [0, 1, 2, True]},
    ],
)
def test_native_selection_rejects_untrusted_bridge_state(
    updates: dict[str, object],
) -> None:
    with pytest.raises(ArtifactError):
        native_process_selection(_NativeRuntime(_state(**updates)), _PROCESSES)


def test_native_physics_axes_remap_representative_reductions() -> None:
    representative = {
        "color_accuracy": "lc",
        "helicities": [{"id": "h:-1,+1"}, {"id": "h:+1,-1"}],
        "color_components": [{"id": "flow:1,2"}, {"id": "flow:2,1"}],
    }
    public = {
        "color_accuracy": "lc",
        "helicities": [{"id": "h:+1,-1"}, {"id": "h:-1,+1"}],
        "color_components": [{"id": "flow:2,1"}, {"id": "flow:1,2"}],
    }
    axes = native_physics_axes(
        _NativeRuntime(_state(), physics=public), representative
    )

    assert axes.public_physics == public
    assert remap_reduction(
        {
            "groups": [
                {
                    "id": "reduction:0",
                    "representative_helicity_id": "h:-1,+1",
                    "representative_color_id": "flow:1,2",
                    "physical_helicity_ids": ["h:-1,+1", "h:+1,-1"],
                    "physical_color_ids": ["flow:1,2", "flow:2,1"],
                }
            ]
        },
        axes,
    ) == {
        "groups": [
            {
                "id": "reduction:0",
                "representative_helicity_id": "h:+1,-1",
                "representative_color_id": "flow:2,1",
                "physical_helicity_ids": ["h:+1,-1", "h:-1,+1"],
                "physical_color_ids": ["flow:2,1", "flow:1,2"],
            }
        ]
    }
    assert representative_vector_to_public((-1, 1), (1, 0)) == (1, -1)


def test_native_physics_axes_reject_missing_reduction_selector() -> None:
    representative = {
        "color_accuracy": "lc",
        "helicities": [{"id": "h:-1"}],
        "color_components": [{"id": "flow:1"}],
    }
    public = {
        "color_accuracy": "lc",
        "helicities": [{"id": "h:+1"}],
        "color_components": [{"id": "flow:1"}],
    }
    axes = native_physics_axes(
        _NativeRuntime(_state(), physics=public), representative
    )

    with pytest.raises(ArtifactError, match="absent from its axis"):
        remap_reduction(
            {
                "groups": [
                    {
                        "representative_helicity_id": "h:missing",
                        "representative_color_id": "flow:1",
                        "physical_helicity_ids": ["h:-1"],
                        "physical_color_ids": ["flow:1"],
                    }
                ]
            },
            axes,
        )


def _particle(index: int, name: str, pdg: int) -> dict[str, object]:
    return {"index": index, "label": index + 1, "particle": name, "pdg": pdg}


def test_process_group_member_physics_returns_to_the_execution_view() -> None:
    representative = {
        "color_accuracy": "lc",
        "external_particles": [
            _particle(0, "u", 2),
            _particle(1, "g", 21),
            _particle(2, "u", 2),
            _particle(3, "g", 21),
        ],
        "helicities": [{"id": "h:+1,+1,+1,+1"}],
        "color_components": [{"id": "flow:3,2,4,1"}, {"id": "flow:3,4,2,1"}],
    }
    public = {
        "process_id": "ubar_g_to_ubar_g",
        "color_accuracy": "lc",
        "external_particles": [
            _particle(0, "u~", -2),
            _particle(1, "g", 21),
            _particle(2, "u~", -2),
            _particle(3, "g", 21),
        ],
        "helicities": [{"id": "h:+1,+1,+1,+1"}],
        "color_components": [
            {
                "kind": "lc-flow",
                "id": "flow:1,4,2,3",
                "word": [1, 4, 2, 3],
                "representative_id": "flow:1,4,2,3",
            },
            {
                "kind": "lc-flow",
                "id": "flow:1,2,4,3",
                "word": [1, 2, 4, 3],
                "representative_id": "flow:1,4,2,3",
            },
        ],
        "reduction": {
            "groups": [
                {
                    "representative_color_id": "flow:1,4,2,3",
                    "physical_color_ids": ["flow:1,4,2,3", "flow:1,2,4,3"],
                }
            ]
        },
    }
    state = _state(
        external_permutation=[0, 1, 2, 3],
        process_group_member_of="representative-key",
        process_group_factor={"numerator": 1, "denominator": 1},
        charge_conjugated=True,
    )
    native = _NativeRuntime(state, physics=public)
    relation = native_process_group_relation(native)
    assert relation is not None
    assert (relation.factor, relation.conjugated) == (Fraction(1), True)

    axes = native_physics_axes(native, representative)
    execution = axes.public_physics
    assert [item["pdg"] for item in execution["external_particles"]] == [2, 21, 2, 21]
    assert [item["id"] for item in execution["color_components"]] == [
        "flow:3,2,4,1",
        "flow:3,4,2,1",
    ]
    assert execution["color_components"][1]["word"] == [3, 4, 2, 1]
    assert execution["reduction"]["groups"][0]["physical_color_ids"] == [
        "flow:3,2,4,1",
        "flow:3,4,2,1",
    ]
    assert axes.color_ids == {
        "flow:3,2,4,1": "flow:3,2,4,1",
        "flow:3,4,2,1": "flow:3,4,2,1",
    }
    assert public["external_particles"][0]["pdg"] == -2
    for identifier in ("flow:3,2,4,1", "flow:singlet", "color:contracted"):
        assert charge_conjugate_color_id(charge_conjugate_color_id(identifier)) == (
            identifier
        )
    assert native_process_group_relation(_NativeRuntime(_state())) is None
    with pytest.raises(ArtifactError, match="process-group factor"):
        native_process_group_relation(
            _NativeRuntime(
                {
                    **state,
                    "process_group_factor": {"numerator": 0, "denominator": 1},
                }
            )
        )


class _Executor:
    def __init__(self) -> None:
        self.color_flows: object = None

    def evaluate_resolved(
        self,
        momenta: object,
        *,
        helicities: object,
        color_flows: object,
        precision: int,
    ) -> ResolvedEvaluation:
        del momenta, helicities, precision
        self.color_flows = color_flows
        return ResolvedEvaluation(
            values=(((Decimal("0.1234567890123456789012345"), Decimal(3)),),),
            helicity_ids=("h:+1,+1,+1,+1",),
            color_ids=("flow:3,2,4,1", "flow:3,4,2,1"),
        )


def test_process_group_exact_results_are_scaled_and_relabelled() -> None:
    inner = _Executor()
    adapter = _ProcessGroupExactExecutor(inner, Fraction(16, 3), True)
    result = adapter.evaluate_resolved(
        (),
        helicities=None,
        color_flows=("flow:1,4,2,3",),
        precision=25,
    )

    assert inner.color_flows == ("flow:3,2,4,1",)
    assert result.color_ids == ("flow:1,4,2,3", "flow:1,2,4,3")
    assert result.values[0][0] == (
        Decimal("0.6584362080658436208065840"),
        Decimal(16),
    )
