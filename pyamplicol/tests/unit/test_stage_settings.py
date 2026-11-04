# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from types import SimpleNamespace

from pyamplicol.evaluators import SymbolicaEvaluatorSettings
from pyamplicol.generation.stage_settings import _stage_symbolica_settings


def _stage(*, slot_width: int, stage_kind: str = "current-combine") -> object:
    return SimpleNamespace(
        stage_kind=stage_kind,
        output_length=1024,
        output_slots=(
            SimpleNamespace(
                component_start=0,
                component_stop=slot_width,
                output_start=0,
                output_stop=16 * slot_width,
            ),
        ),
    )


def test_high_rank_current_stage_caps_native_output_chunks_at_256() -> None:
    settings = SymbolicaEvaluatorSettings(compiled_output_chunk_size=512)

    effective = _stage_symbolica_settings(
        _stage(slot_width=16),
        None,
        settings,
    )

    assert effective.compiled_output_chunk_size == 256


def test_standard_current_and_amplitude_stages_retain_requested_chunk_size() -> None:
    settings = SymbolicaEvaluatorSettings(compiled_output_chunk_size=512)

    vector = _stage_symbolica_settings(_stage(slot_width=4), None, settings)
    auxiliary_tensor = _stage_symbolica_settings(
        _stage(slot_width=6),
        None,
        settings,
    )
    amplitude = _stage_symbolica_settings(
        _stage(slot_width=16, stage_kind="amplitude"),
        None,
        settings,
    )

    assert vector.compiled_output_chunk_size == 512
    assert auxiliary_tensor.compiled_output_chunk_size == 512
    assert amplitude.compiled_output_chunk_size == 512
