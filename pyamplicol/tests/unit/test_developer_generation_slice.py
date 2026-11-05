# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from types import MappingProxyType

import pytest

from pyamplicol.config import ProcessConfig, RunConfig
from pyamplicol.generation.service import GenerationBackend
from tools.developer.generation_slice import GenerationSlice


def test_generation_slice_normalizes_immutable_internal_selection() -> None:
    generation_slice = GenerationSlice(
        reference_color_order=(2, 4, 1, 3),
        selected_color_sector_ids=(3, 1),
        selected_source_helicities={1: -1, 2: 1},
    )

    selection = generation_slice._selection()

    assert selection.reference_color_order == (2, 4, 1, 3)
    assert selection.selected_color_sector_ids == frozenset({1, 3})
    assert selection.selected_source_helicities == {1: -1, 2: 1}
    assert isinstance(selection.selected_source_helicities, MappingProxyType)


def test_generation_backend_private_slice_overrides_legacy_public_fields() -> None:
    backend = GenerationBackend(
        RunConfig(
            action="generate",
            process=ProcessConfig(
                reference_color_order=(1, 2, 3, 4),
                selected_color_sector_ids=(0,),
                selected_source_helicities={"1": 1},
            ),
        ),
        None,
        process_selection=GenerationSlice(
            reference_color_order=(2, 4, 1, 3),
            selected_color_sector_ids=(2,),
            selected_source_helicities={1: -1},
        )._selection(),
    )

    selection = backend._process_selection

    assert selection.reference_color_order == (2, 4, 1, 3)
    assert selection.selected_color_sector_ids == frozenset({2})
    assert selection.selected_source_helicities == {1: -1}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"reference_color_order": (1, 1)}, "unique labels"),
        ({"selected_color_sector_ids": (-1,)}, "non-negative"),
        ({"selected_source_helicities": {0: 1}}, "positive integer labels"),
        ({"selected_source_helicities": {1: True}}, "integer helicities"),
    ),
)
def test_generation_slice_rejects_invalid_private_selectors(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        GenerationSlice(**kwargs)  # type: ignore[arg-type]
