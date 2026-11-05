# SPDX-License-Identifier: 0BSD
"""Automatic FFT uses the existing support verdict, never an exception catch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pyamplicol.color import ColorContractionPlan, build_color_plan
from pyamplicol.generation import recurrence_physics as physics
from pyamplicol.generation import service
from pyamplicol.models.builtin.process_ir import build_process_ir


def _contract(lane: str, method: str):
    process = build_process_ir("g g > g g", color_accuracy="full")
    plan = build_color_plan(process, color_accuracy="full")
    if lane == "on-the-fly":
        return physics.build_on_the_fly_color_contraction(plan, contraction=method)[0]
    labels = tuple(leg.label for leg in process.legs)
    logical = SimpleNamespace(
        layout="contracted-color-union",
        replay_partitions=(),
        external_legs=tuple(
            SimpleNamespace(source_slot=slot, public_label=label)
            for slot, label in enumerate(labels)
        ),
        physical_sectors=tuple(
            SimpleNamespace(
                sector_id=sector.id,
                kind=sector.kind,
                word_source_slots=tuple(
                    labels.index(label) for label in sector.color_words[0]
                ),
            )
            for sector in plan.sectors
        ),
    )
    return physics.build_recurrence_color_contraction(
        logical,
        plan,
        ((-1, -1, 1, 1),),
        tuple((sector.id, 0) for sector in plan.sectors),
        (),
        contraction=method,
    )


@pytest.mark.parametrize("lane", ("recurrence", "on-the-fly"))
def test_automatic_contraction_selects_a_supported_fft_plan(lane):
    automatic = _contract(lane, "auto")
    forced = _contract(lane, "symmetric-group-fft")
    assert automatic == forced
    assert automatic.symmetric_group_block is not None


@pytest.mark.parametrize("lane", ("recurrence", "on-the-fly"))
def test_only_automatic_contraction_falls_back_on_unsupported_fft(lane, monkeypatch):
    calls = []

    def unsupported(plan, groups, **_kwargs):
        calls.append(True)
        return ColorContractionPlan(
            color_accuracy=plan.color_accuracy,
            supported=False,
            reason="test: no certified permutation orbit",
            group_count=len(groups),
            entries=(),
        )

    monkeypatch.setattr(
        physics, "build_symmetric_group_color_contraction_plan", unsupported
    )
    direct = _contract(lane, "direct")
    assert not calls  # The explicit opt-out does not even probe FFT support.
    assert _contract(lane, "auto") == direct
    with pytest.raises(ValueError, match="no certified permutation orbit"):
        _contract(lane, "symmetric-group-fft")
    assert len(calls) == 2
    if lane == "on-the-fly":
        process = build_process_ir("g g > g g", color_accuracy="full")
        plan = build_color_plan(process, color_accuracy="full")
        payload = service._build_on_the_fly_contracted_color_payload_v1(
            plan, contraction="auto"
        )
        assert payload.summary["storage"] == "expanded"
        assert payload.summary["factorization"] is None
        assert "fft_provenance" not in payload.summary


@pytest.mark.parametrize("lane", ("recurrence", "on-the-fly"))
def test_automatic_contraction_does_not_hide_unrelated_failures(lane, monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("unexpected FFT construction failure")

    monkeypatch.setattr(physics, "build_symmetric_group_color_contraction_plan", broken)
    with pytest.raises(RuntimeError, match="unexpected FFT construction failure"):
        _contract(lane, "auto")
