# SPDX-License-Identifier: 0BSD
"""Performance and parity contracts for eager helicity filtering."""

from __future__ import annotations

import pytest

from pyamplicol.generation import dag_algorithms
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.dag_types import GenericDAG
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir


@pytest.fixture(scope="module")
def three_line_nlc_dag() -> tuple[BuiltinSMModel, GenericDAG]:
    model = BuiltinSMModel()
    dag = compile_generic_dag(
        build_process_ir("d d~ > u u~ s s~ g", color_accuracy="nlc"),
        model=model,
        online_evaluation_reuse=True,
    )
    return model, dag


def test_compact_helicity_filter_exactly_matches_generic_fallback(
    three_line_nlc_dag: tuple[BuiltinSMModel, GenericDAG],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, dag = three_line_nlc_dag
    compact = dag_algorithms.prune_global_helicity_flip_equivalent_roots(dag, model)

    monkeypatch.setattr(
        dag_algorithms,
        "_compact_helicity_flip_representatives",
        lambda *args, **kwargs: None,
    )
    generic = dag_algorithms.prune_global_helicity_flip_equivalent_roots(dag, model)

    assert compact == generic
    assert compact.currents == generic.currents
    assert compact.interactions == generic.interactions
    assert compact.amplitude_roots == generic.amplitude_roots


def test_compact_helicity_filter_does_not_scan_all_decorated_sources_per_root(
    three_line_nlc_dag: tuple[BuiltinSMModel, GenericDAG],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, dag = three_line_nlc_dag
    assert len(dag.sources) > 1_000
    assert {
        int(
            dag.currents[root.left_id].index.helicity_ancestry
            | dag.currents[root.right_id].index.helicity_ancestry
        ).bit_count()
        for root in dag.amplitude_roots
    } == {len(dag.process.legs)}

    def reject_generic_scan(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("compact filtering fell back to all-source scanning")

    monkeypatch.setattr(
        dag_algorithms,
        "_root_physical_helicity_signature",
        reject_generic_scan,
    )

    reduced = dag_algorithms.prune_global_helicity_flip_equivalent_roots(dag, model)

    assert len(reduced.amplitude_roots) == len(dag.amplitude_roots) // 2
