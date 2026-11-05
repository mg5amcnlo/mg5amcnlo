# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from tools.developer.runtime_selector_profile import selector_pattern


def test_selector_patterns_are_deterministic_stable_grouping_inputs() -> None:
    selectors = ("a", "b", "c", "d")
    assert selector_pattern("homogeneous", selectors, 8, seed=7) == ("a",) * 8
    assert selector_pattern("pre-pooled", selectors, 10, seed=7) == (
        "a",
        "a",
        "a",
        "b",
        "b",
        "b",
        "c",
        "c",
        "d",
        "d",
    )
    assert selector_pattern("alternating", selectors, 8, seed=7) == selectors * 2
    random_first = selector_pattern("seeded-random", selectors, 32, seed=7)
    random_second = selector_pattern("seeded-random", selectors, 32, seed=7)
    assert random_first == random_second
    assert set(random_first) == set(selectors)
    assert random_first != selector_pattern("seeded-random", selectors, 32, seed=8)
