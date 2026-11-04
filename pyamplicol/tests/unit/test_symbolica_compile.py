# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import pyamplicol.evaluators.symbolica_compile as symbolica_compile
from pyamplicol.evaluators.symbolica_settings import SymbolicaEvaluatorSettings


@pytest.mark.parametrize("merge", [False, True])
@pytest.mark.parametrize("chunk_size", [None, 1])
def test_named_functions_are_inlined_with_tags_and_closed_parameters(
    monkeypatch: pytest.MonkeyPatch, merge: bool, chunk_size: int | None
) -> None:
    from symbolica import S

    x, unused, closed, argument, function, tagged = S(
        "compile_x",
        "compile_unused",
        "compile_closed",
        "compile_argument",
        "compile_function",
        "compile_tagged",
    )
    finalized: list[Any] = []
    finalize = symbolica_compile._finalize_symbolica_evaluator

    def record_finalized(evaluator: Any, *args: Any, **kwargs: Any) -> Any:
        finalized.append(evaluator)
        return finalize(evaluator, *args, **kwargs)

    monkeypatch.setattr(
        symbolica_compile, "_finalize_symbolica_evaluator", record_finalized
    )
    evaluator = symbolica_compile._compile_symbolica_outputs(
        (function(x), tagged(7, x) + 1),
        [x, unused, closed],
        functions={
            (function, (argument,)): argument**2 + closed,
            (tagged(7), (argument,)): function(argument) + argument,
        },
        merge_evaluators_strategy=merge,
        verbose_evaluator_build=False,
        jit_compile=False,
        symbolica_settings=SymbolicaEvaluatorSettings(
            iterations=1,
            n_cores=1,
            compiled_output_chunk_size=chunk_size,
        ),
    )
    rows = np.asarray([[2, 99, 3], [1 + 2j, -11, 4]], dtype=np.complex128)
    actual = evaluator.evaluate_complex(rows)
    expected_first = rows[:, 0] ** 2 + rows[:, 2]
    expected = np.column_stack((expected_first, expected_first + rows[:, 0] + 1))

    np.testing.assert_array_equal(actual, expected)
    assert len(finalized) == (1 if chunk_size is None else 2)
    for compiled in finalized:
        assert compiled.get_instructions().sub_evaluators == []
