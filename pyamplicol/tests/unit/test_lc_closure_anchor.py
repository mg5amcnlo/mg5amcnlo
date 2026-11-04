# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from pyamplicol.color import build_color_plan
from pyamplicol.generation.dag_algorithms import (
    _canonical_open_line_alias_owner_sector_ids,
    _canonical_open_line_alias_owner_sector_map,
    _physical_sink_sector_ids,
    _retain_canonical_open_line_alias_roots,
    infer_minimal_coupling_order_limits,
    prune_dag_to_amplitude_roots,
)
from pyamplicol.generation.dag_color import ColorEngine
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.dag_ordering import _closure_candidate_splits
from pyamplicol.generation.dag_types import AmplitudeRoot, GenericDAG
from pyamplicol.generation.helicity_replay import _SemanticAmplitudeRootSignature
from pyamplicol.models._physics_ir import ContractionIR
from pyamplicol.models.builtin.model import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir


def _sector_for_word(expression: str, word: tuple[int, ...]):
    process = build_process_ir(expression)
    plan = build_color_plan(process)
    return process, next(
        sector for sector in plan.sectors if sector.word_labels == word
    )


def _alias_root_test_dag(
    sector_ids: tuple[int, ...],
) -> GenericDAG:
    process = build_process_ir(
        "d d~ > t t~ g g",
        color_accuracy="full",
    )
    plan = build_color_plan(process, color_accuracy="full")
    contraction = ContractionIR(
        name="alias-root-test",
        left_basis="scalar",
        right_basis="scalar",
        coefficients=((1.0, 0.0),),
    )
    roots = tuple(
        AmplitudeRoot(
            id=root_id,
            kind="direct",
            left_id=root_id,
            right_id=root_id,
            color_weight=(1.0, 0.0),
            contraction_ir=contraction,
            color_sector_id=sector_id,
        )
        for root_id, sector_id in enumerate(sector_ids)
    )
    return GenericDAG(
        process=process,
        color_plan=plan,
        currents=(),
        sources=(),
        interactions=(),
        amplitude_roots=roots,
    )


def _first_open_line_alias_pair(dag: GenericDAG) -> tuple[int, int]:
    owner_by_sector = _canonical_open_line_alias_owner_sector_map(dag.color_plan)
    aliases_by_owner: dict[int, list[int]] = {}
    for sector_id, owner_id in owner_by_sector.items():
        aliases_by_owner.setdefault(owner_id, []).append(sector_id)
    pair = next(
        tuple(sorted(sector_ids))[:2]
        for sector_ids in aliases_by_owner.values()
        if len(sector_ids) >= 2
    )
    assert len(pair) == 2
    return pair[0], pair[1]


def _alias_signature(label: str) -> _SemanticAmplitudeRootSignature:
    return _SemanticAmplitudeRootSignature(
        proof_digest=label * 64,
        selector_states=(),
        factor=(1.0, 0.0),
    )


def test_proof_identical_aliases_reject_unequal_root_multiplicity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = _first_open_line_alias_pair(_alias_root_test_dag(()))
    dag = _alias_root_test_dag((first, first, second))
    signature = _alias_signature("a")
    monkeypatch.setattr(
        "pyamplicol.generation.dag_algorithms."
        "_semantic_amplitude_root_alias_signatures",
        lambda *_args, **_kwargs: (signature, signature, signature),
    )

    with pytest.raises(ValueError, match="unequal root multiplicities"):
        _retain_canonical_open_line_alias_roots(dag, BuiltinSMModel())


def test_proof_identical_aliases_with_equal_multiplicity_still_collapse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = _first_open_line_alias_pair(_alias_root_test_dag(()))
    dag = _alias_root_test_dag((first, first, second, second))
    signature = _alias_signature("a")
    monkeypatch.setattr(
        "pyamplicol.generation.dag_algorithms."
        "_semantic_amplitude_root_alias_signatures",
        lambda *_args, **_kwargs: (signature,) * 4,
    )

    retained = _retain_canonical_open_line_alias_roots(dag, BuiltinSMModel())

    owner = _canonical_open_line_alias_owner_sector_map(dag.color_plan)[first]
    assert len(retained.amplitude_roots) == 2
    assert {root.color_sector_id for root in retained.amplitude_roots} == {owner}


def test_distinct_alias_proof_signatures_remain_a_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = _first_open_line_alias_pair(_alias_root_test_dag(()))
    dag = _alias_root_test_dag((first, first, second, second))
    signature_a = _alias_signature("a")
    signature_b = _alias_signature("b")
    monkeypatch.setattr(
        "pyamplicol.generation.dag_algorithms."
        "_semantic_amplitude_root_alias_signatures",
        lambda *_args, **_kwargs: (
            signature_a,
            signature_b,
            signature_a,
            signature_b,
        ),
    )

    retained = _retain_canonical_open_line_alias_roots(dag, BuiltinSMModel())

    assert len(retained.amplitude_roots) == 2
    assert {root.left_id for root in retained.amplitude_roots} == {0, 1}


def test_two_line_closure_rotates_public_blocks_from_source_slot_zero() -> None:
    process, sector = _sector_for_word("d d~ > t t~ g g", (2, 5, 6, 4, 3, 1))

    assert sector.canonical_closure_traversal_word(process) == (3, 1, 2, 5, 6, 4)
    assert sector.canonical_closure_sink_label(process) == 4
    assert sector.word_labels == (2, 5, 6, 4, 3, 1)


def test_public_reference_selects_sector_without_overriding_private_closure() -> None:
    process, sector = _sector_for_word(
        "d d~ > t t~ g g",
        (2, 5, 6, 4, 3, 1),
    )
    model = BuiltinSMModel()
    plan = build_color_plan(process)
    limits = infer_minimal_coupling_order_limits(
        process,
        model=model,
        selected_color_sector_ids=(sector.id,),
    )
    dag = prune_dag_to_amplitude_roots(
        compile_generic_dag(
            process,
            model=model,
            color_plan=plan,
            reference_color_order=sector.word_labels,
            selected_color_sector_ids=(sector.id,),
            max_coupling_orders=limits,
            closure_side_mask_pruning=False,
            color_order_mask_pruning=False,
            species_reachability_pruning=False,
        )
    )

    assert plan.sectors[sector.id].word_labels == (2, 5, 6, 4, 3, 1)
    assert (
        len(dag.currents),
        len(dag.interactions),
        len(dag.amplitude_roots),
        sum(
            current.index.external_mask.bit_count() == 1
            for current in dag.currents
        ),
    ) == (78, 150, 32, 12)


def test_closure_reconstructs_blocks_independently_of_line_tuple_order() -> None:
    process, sector = _sector_for_word("d d~ > t t~ g", (2, 5, 4, 3, 1))
    reordered = replace(
        sector,
        open_color_lines=tuple(reversed(sector.open_color_lines)),
    )

    assert reordered.canonical_closure_traversal_word(process) == (3, 1, 2, 5, 4)
    assert reordered.canonical_closure_sink_label(process) == 4


def test_closure_uses_source_position_not_numeric_public_label() -> None:
    process, sector = _sector_for_word("d d~ > t t~", (2, 4, 3, 1))
    label_map = {1: 10, 2: 5, 3: 20, 4: 30}
    relabelled_process = replace(
        process,
        legs=tuple(
            replace(leg, label=label_map[leg.label])
            for leg in process.legs
        ),
    )
    relabelled_sector = replace(
        sector,
        word_labels=tuple(label_map[label] for label in sector.word_labels),
        open_color_lines=tuple(
            replace(
                line,
                fundamental_label=label_map[line.fundamental_label],
                antifundamental_label=label_map[line.antifundamental_label],
                adjoint_labels=tuple(
                    label_map[label] for label in line.adjoint_labels
                ),
            )
            for line in sector.open_color_lines
        ),
    )

    assert relabelled_process.legs[0].label == 10
    assert min(label_map.values()) == 5
    assert relabelled_sector.canonical_closure_traversal_word(
        relabelled_process
    ) == (20, 10, 5, 30)


def test_singlet_source_zero_selects_minimum_coloured_source_slot() -> None:
    process = build_process_ir("e- d > e- d t t~")
    plan = build_color_plan(process)
    sector = next(
        sector
        for sector in plan.sectors
        if sector.word_labels == (4, 6, 5, 2)
    )

    assert process.legs[0].is_singlet
    assert process.legs[1].label == 2
    assert sector.canonical_closure_traversal_word(process) == (5, 2, 4, 6)
    assert sector.canonical_closure_sink_label(process) == 6


def test_single_trace_and_singlet_sectors_keep_public_words() -> None:
    for expression in ("g g > g g", "z z > z z"):
        process = build_process_ir(expression)
        plan = build_color_plan(process)
        assert all(
            sector.canonical_closure_traversal_word(process)
            == sector.color_words[0]
            for sector in plan.sectors
        )


def test_non_lc_open_line_roots_use_the_canonical_private_sink() -> None:
    model = BuiltinSMModel()
    for accuracy in ("nlc", "full"):
        process = build_process_ir(
            "d d~ > t t~ g g",
            color_accuracy=accuracy,
        )
        plan = build_color_plan(process, color_accuracy=accuracy)
        engine = ColorEngine(plan, model)
        candidate_splits = _closure_candidate_splits(process, model, engine)

        assert candidate_splits == ((55, 8),)
        assert _physical_sink_sector_ids(plan, candidate_splits) == frozenset(
            sector.id for sector in plan.sectors
        )

        owner_sector_ids = frozenset(range(0, 24, 2))
        assert _canonical_open_line_alias_owner_sector_ids(
            plan,
            (sector.id for sector in plan.sectors),
        ) == owner_sector_ids
        complete_limits = infer_minimal_coupling_order_limits(
            process,
            model=model,
        )
        complete_dag = compile_generic_dag(
            process,
            model=model,
            color_plan=plan,
            max_coupling_orders=complete_limits,
        )
        assert len(complete_dag.amplitude_roots) == 384
        assert {
            root.color_sector_id for root in complete_dag.amplitude_roots
        } == owner_sector_ids
        assert all(
            sum(
                root.color_sector_id == sector_id
                for root in complete_dag.amplitude_roots
            )
            == 32
            for sector_id in owner_sector_ids
        )
        backward_dag = compile_generic_dag(
            process,
            model=model,
            color_plan=plan,
            max_coupling_orders=complete_limits,
            backward_live_planning=True,
        )
        assert {
            root.color_sector_id for root in backward_dag.amplitude_roots
        } == owner_sector_ids

        # A restricted non-owner alias becomes the sole active owner rather
        # than being erased by ownership derived from the complete plan.
        sector = plan.sectors[13]
        limits = infer_minimal_coupling_order_limits(
            process,
            model=model,
            selected_color_sector_ids=(sector.id,),
        )
        dag = compile_generic_dag(
            process,
            model=model,
            color_plan=plan,
            selected_color_sector_ids=(sector.id,),
            max_coupling_orders=limits,
        )
        traversal = sector.canonical_closure_traversal_word(process)

        assert len(dag.amplitude_roots) == 32
        assert {
            dag.currents[root.right_id].index.external_labels
            for root in dag.amplitude_roots
        } == {(traversal[-1],)}
        assert {
            dag.currents[root.left_id].index.ordered_external_labels
            for root in dag.amplitude_roots
        } == {traversal[:-1]}
        assert any(
            current.index.particle_id == 6
            and current.index.external_mask.bit_count() > 1
            for current in dag.currents
        )
        assert not any(
            current.index.particle_id == -6
            and current.index.external_mask.bit_count() > 1
            for current in dag.currents
        )
        root_keys = {
            (
                root.kind,
                root.left_id,
                root.right_id,
                root.color_sector_id,
                root.vertex_kind,
                root.vertex_particles,
                root.contraction_ir,
            )
            for root in dag.amplitude_roots
        }
        assert len(root_keys) == len(dag.amplitude_roots)


def test_three_open_line_private_sinks_are_resolved_per_sector() -> None:
    model = BuiltinSMModel()
    full_mask = (1 << 6) - 1
    for accuracy in ("nlc", "full"):
        process = build_process_ir(
            "d d~ > u u~ s s~",
            color_accuracy=accuracy,
        )
        plan = build_color_plan(process, color_accuracy=accuracy)
        engine = ColorEngine(plan, model)

        assert len(plan.sectors) == 36
        assert _canonical_open_line_alias_owner_sector_ids(
            plan,
            (sector.id for sector in plan.sectors),
        ) == frozenset({0, 6, 12, 18, 24, 30})

        assert {
            sector.canonical_closure_sink_label(process) for sector in plan.sectors
        } == {4, 6}
        for sink_label in (4, 6):
            sink_mask = 1 << (sink_label - 1)
            expected_sector_ids = frozenset(
                sector.id
                for sector in plan.sectors
                if sector.canonical_closure_sink_label(process) == sink_label
            )
            assert (
                _physical_sink_sector_ids(
                    plan,
                    ((full_mask ^ sink_mask, sink_mask),),
                )
                == expected_sector_ids
            )

        assert {
            right_mask
            for _left_mask, right_mask in _closure_candidate_splits(
                process,
                model,
                engine,
            )
        } == {1 << 3, 1 << 5}

        limits = infer_minimal_coupling_order_limits(process, model=model)
        dag = compile_generic_dag(
            process,
            model=model,
            color_plan=plan,
            max_coupling_orders=limits,
        )
        # The public tensor owners remain stable, including rootless sector 24.
        # Semantic closure ownership retains both distinct traversal families
        # where they exist instead of deleting an entire alias sector.
        root_owner_ids = frozenset({0, 6, 12, 18, 24, 30})
        assert len(dag.amplitude_roots) == 80
        assert {root.color_sector_id for root in dag.amplitude_roots} == root_owner_ids
        assert Counter(
            root.color_sector_id for root in dag.amplitude_roots
        ) == Counter({0: 16, 6: 16, 12: 16, 18: 8, 24: 8, 30: 16})
        assert {
            dag.currents[root.right_id].index.external_labels
            for root in dag.amplitude_roots
            if root.color_sector_id == 0
        } == {(4,), (6,)}
        backward_dag = compile_generic_dag(
            process,
            model=model,
            color_plan=plan,
            max_coupling_orders=limits,
            backward_live_planning=True,
        )
        assert {
            root.color_sector_id for root in backward_dag.amplitude_roots
        } == root_owner_ids
        assert len(backward_dag.amplitude_roots) == len(dag.amplitude_roots)


@pytest.mark.parametrize("accuracy", ("nlc", "full"))
def test_identical_three_line_aliases_preserve_exchange_closures(
    accuracy: str,
) -> None:
    model = BuiltinSMModel()
    process = build_process_ir(
        "d d~ > u u~ u u~",
        color_accuracy=accuracy,
    )
    limits = infer_minimal_coupling_order_limits(process, model=model)

    dag = compile_generic_dag(
        process,
        model=model,
        max_coupling_orders=limits,
    )
    backward_dag = compile_generic_dag(
        process,
        model=model,
        max_coupling_orders=limits,
        backward_live_planning=True,
    )

    expected_by_owner = Counter({0: 24, 6: 24, 12: 20, 18: 20, 24: 20, 30: 20})
    assert Counter(root.color_sector_id for root in dag.amplitude_roots) == (
        expected_by_owner
    )
    assert Counter(
        root.color_sector_id for root in backward_dag.amplitude_roots
    ) == expected_by_owner
    assert all(
        {
            dag.currents[root.right_id].index.external_labels
            for root in dag.amplitude_roots
            if root.color_sector_id == owner
        }
        == {(4,), (6,)}
        for owner in expected_by_owner
    )


@pytest.mark.parametrize("accuracy", ("nlc", "full"))
def test_four_open_line_aliases_retain_semantic_closure_union(
    accuracy: str,
) -> None:
    model = BuiltinSMModel()
    process = build_process_ir(
        "d d~ > u u~ s s~ c c~",
        color_accuracy=accuracy,
    )
    limits = infer_minimal_coupling_order_limits(process, model=model)
    dag = compile_generic_dag(
        process,
        model=model,
        max_coupling_orders=limits,
        online_evaluation_reuse=True,
        backward_live_planning=True,
    )

    counts = Counter(root.color_sector_id for root in dag.amplitude_roots)
    assert len(dag.color_plan.sectors) == 576
    assert len(counts) == 24
    assert len(dag.amplitude_roots) == 672
    assert set(counts.values()) == {8, 24, 32, 48}
    assert all(sector_id % 24 == 0 for sector_id in counts)


def test_full_colour_pure_gluon_closure_remains_on_public_sinks() -> None:
    process = build_process_ir("g g > g g", color_accuracy="full")
    plan = build_color_plan(process, color_accuracy="full")
    model = BuiltinSMModel()
    candidate_splits = _closure_candidate_splits(
        process,
        model,
        ColorEngine(plan, model),
    )

    assert all(
        sector.canonical_closure_sink_label(process) == sector.color_words[0][-1]
        for sector in plan.sectors
    )
    assert _physical_sink_sector_ids(plan, candidate_splits) == frozenset(
        sector.id for sector in plan.sectors
    )
    assert _canonical_open_line_alias_owner_sector_ids(
        plan,
        (sector.id for sector in plan.sectors),
    ) == frozenset(sector.id for sector in plan.sectors)
    dag = compile_generic_dag(process, model=model, color_plan=plan)
    assert len(dag.amplitude_roots) == 96
    assert Counter(root.color_sector_id for root in dag.amplitude_roots) == Counter(
        {sector.id: 16 for sector in plan.sectors}
    )
    with pytest.raises(ValueError, match="absent from the color plan: 999"):
        _canonical_open_line_alias_owner_sector_ids(plan, (999,))
