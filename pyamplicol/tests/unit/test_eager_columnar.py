# SPDX-License-Identifier: 0BSD
"""Contracts for the Python-to-Rust eager lowering input."""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
import pytest

from pyamplicol.color import (
    build_color_contraction_plan,
    build_lc_topology_replay_plan,
)
from pyamplicol.color.plan import (
    build_color_plan,
    build_color_topology_replay_certificate,
)
from pyamplicol.generation.dag_algorithms import (
    prune_global_helicity_flip_equivalent_roots,
)
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.dag_types import GenericDAG
from pyamplicol.generation.eager_columnar import (
    EAGER_LOWERING_INPUT_ABI,
    EAGER_LOWERING_INPUT_SCHEMA_SHA256,
    FACTOR_EXACT_SOURCE_BINARY64,
    FACTOR_EXACT_SOURCE_CANONICAL_IR,
    EagerColumn,
    EagerColumnarInputError,
    EagerColumnarTable,
    EagerLoweringInputV1,
    build_eager_lowering_input_v1,
)
from pyamplicol.generation.eager_lowering import MappingEagerKernelResolver
from pyamplicol.generation.helicity_materialization import (
    materialize_helicity_recurrence,
)
from pyamplicol.generation.helicity_replay import build_helicity_recurrence_plan
from pyamplicol.generation.runtime_amplitudes import _amplitude_groups
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.base import Model
from pyamplicol.models.builtin.process_ir import build_process_ir


@dataclass(frozen=True, slots=True)
class _Case:
    model: Model
    dag: GenericDAG
    resolver: MappingEagerKernelResolver


def _resolver_for(
    dag: GenericDAG,
    model: Model,
    *,
    reverse: bool = False,
) -> MappingEagerKernelResolver:
    vertex_items = [
        (kind, 100 + index)
        for index, kind in enumerate(sorted(dag.required_vertex_kinds))
    ]
    propagator_keys: set[tuple[int, int]] = set()
    for current in dag.currents:
        propagator = model._propagator_ir(
            current.index.particle_id,
            current.index.chirality,
        )
        if propagator.applies_propagator:
            propagator_keys.add((current.index.particle_id, current.index.chirality))
    propagator_items = [
        (key, 1_000 + index) for index, key in enumerate(sorted(propagator_keys))
    ]
    closure_items = [
        ((str(root.kind), root.vertex_kind), 2_000 + root.id)
        for root in dag.amplitude_roots
        if root.kind != "direct-contraction"
    ]
    if reverse:
        vertex_items.reverse()
        propagator_items.reverse()
        closure_items.reverse()
    return MappingEagerKernelResolver(
        vertex_kernels=dict(vertex_items),
        propagator_kernels=dict(propagator_items),
        closure_kernels=dict(closure_items),
    )


def _case(
    expression: str = "d d~ > z g",
    *,
    recurrence: bool = False,
    prepared_closure: bool = True,
) -> _Case:
    model = BuiltinSMModel()
    dag = compile_generic_dag(build_process_ir(expression), model=model)
    if recurrence:
        reduced = prune_global_helicity_flip_equivalent_roots(dag, model)
        proof = build_helicity_recurrence_plan(reduced, model)
        assert proof is not None
        materialization = materialize_helicity_recurrence(reduced, proof)
        dag = replace(
            materialization.dag,
            helicity_recurrence=proof,
            helicity_materialization=materialization,
        )
    if prepared_closure:
        first = replace(
            dag.amplitude_roots[0],
            kind="prepared-closure",
            vertex_kind=3,
            vertex_particles=(21, 21, 21),
            coupling=(1.0, 0.0),
        )
        dag = replace(dag, amplitude_roots=(first, *dag.amplitude_roots[1:]))
    return _Case(model, dag, _resolver_for(dag, model))


@pytest.fixture(scope="module")
def low_case() -> _Case:
    return _case()


@pytest.fixture(scope="module")
def lowering_input(low_case: _Case) -> EagerLoweringInputV1:
    return build_eager_lowering_input_v1(
        dag=low_case.dag,
        model=low_case.model,
        resolver=low_case.resolver,
        process_id="columnar_contract",
    )


def _table_bytes(value: EagerLoweringInputV1) -> tuple[object, ...]:
    return tuple(
        (
            table.name,
            table.row_count,
            tuple(
                (
                    column.name,
                    column.values.dtype.str,
                    column.values.shape,
                    column.values.tobytes(order="C"),
                )
                for column in table.columns
            ),
        )
        for table in value.tables
    )


def _f64_bits(value: float) -> str:
    return f"{int.from_bytes(struct.pack('<d', value), 'little'):016x}"


def test_contract_is_flat_immutable_little_endian_and_complete(
    lowering_input: EagerLoweringInputV1,
) -> None:
    assert lowering_input.abi == EAGER_LOWERING_INPUT_ABI
    assert tuple(table.name for table in lowering_input.tables) == tuple(
        sorted(table.name for table in lowering_input.tables)
    )
    expected = {
        "currents",
        "sources",
        "interactions",
        "roots",
        "contraction_coefficients",
        "couplings",
        "model_parameters",
        "helicity_selectors",
        "color_selectors",
        "coherent_groups",
        "reduction_members",
        "color_contraction_entries",
        "exact_factors",
        "momentum_masks",
        "bitset_ranges",
        "bitset_words",
    }
    assert expected <= {table.name for table in lowering_input.tables}
    assert not isinstance(lowering_input.string_catalog, list)
    assert not isinstance(lowering_input.canonical_ir_catalog, list)
    assert not any(isinstance(value, Mapping) for value in lowering_input.tables)

    for table in lowering_input.tables:
        for column in table.columns:
            values = column.values
            assert values.dtype != np.dtype("O")
            assert values.flags.c_contiguous
            assert not values.flags.writeable
            assert values.dtype.byteorder in {"<", "|", "="}

    assert lowering_input.schema_sha256 == EAGER_LOWERING_INPUT_SCHEMA_SHA256


def test_contract_bytes_and_digest_ignore_resolver_mapping_order(
    low_case: _Case,
    lowering_input: EagerLoweringInputV1,
) -> None:
    reordered = build_eager_lowering_input_v1(
        dag=low_case.dag,
        model=low_case.model,
        resolver=_resolver_for(low_case.dag, low_case.model, reverse=True),
        process_id="columnar_contract",
    )

    assert reordered.digest == lowering_input.digest
    assert reordered.string_catalog == lowering_input.string_catalog
    assert reordered.canonical_ir_catalog == lowering_input.canonical_ir_catalog
    assert _table_bytes(reordered) == _table_bytes(lowering_input)


def test_contract_rejects_an_out_of_range_exact_ir_reference(
    lowering_input: EagerLoweringInputV1,
) -> None:
    factors = lowering_input.table("exact_factors")
    bad_values = factors.column("exact_ir_id").copy()
    bad_values[0] = len(lowering_input.canonical_ir_catalog)
    bad_values.flags.writeable = False
    bad_columns = tuple(
        EagerColumn(column.name, bad_values) if column.name == "exact_ir_id" else column
        for column in factors.columns
    )
    bad_table = EagerColumnarTable(factors.name, factors.row_count, bad_columns)
    tables = tuple(
        bad_table if table.name == factors.name else table
        for table in lowering_input.tables
    )

    with pytest.raises(EagerColumnarInputError, match="absent row"):
        EagerLoweringInputV1(
            abi=lowering_input.abi,
            process_key=lowering_input.process_key,
            model_name=lowering_input.model_name,
            string_catalog=lowering_input.string_catalog,
            canonical_ir_catalog=lowering_input.canonical_ir_catalog,
            tables=tables,
        )


def test_extraction_never_calls_runtime_schema_layout(
    monkeypatch: pytest.MonkeyPatch,
    low_case: _Case,
) -> None:
    from pyamplicol.generation import runtime_schema

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("column extraction called runtime-schema construction")

    monkeypatch.setattr(runtime_schema, "build_runtime_schema_layout", forbidden)
    result = build_eager_lowering_input_v1(
        dag=low_case.dag,
        model=low_case.model,
        resolver=low_case.resolver,
    )
    assert result.table("interactions").row_count == len(low_case.dag.interactions)


def test_checked_u32_rejects_an_overflowing_dag_id(low_case: _Case) -> None:
    overflowing = replace(low_case.dag.currents[0], id=1 << 32)
    dag = replace(low_case.dag, currents=(overflowing, *low_case.dag.currents[1:]))

    with pytest.raises(EagerColumnarInputError, match="does not fit u32"):
        build_eager_lowering_input_v1(
            dag=dag,
            model=low_case.model,
            resolver=_resolver_for(dag, low_case.model),
        )


def test_arbitrary_width_masks_are_flat_u64_words(low_case: _Case) -> None:
    high_mask = (1 << 130) | (1 << 65) | 1
    index = replace(
        low_case.dag.currents[0].index,
        momentum_mask=high_mask,
        helicity_ancestry=high_mask,
    )
    current = replace(low_case.dag.currents[0], index=index)
    dag = replace(low_case.dag, currents=(current, *low_case.dag.currents[1:]))
    result = build_eager_lowering_input_v1(
        dag=dag,
        model=low_case.model,
        resolver=_resolver_for(dag, low_case.model),
    )

    current_table = result.table("currents")
    bitset_id = int(current_table.column("momentum_mask_bitset_id")[0])
    ranges = result.table("bitset_ranges")
    words = result.table("bitset_words").column("value")
    start = int(ranges.column("start")[bitset_id])
    count = int(ranges.column("count")[bitset_id])
    rebuilt = sum(int(words[start + index]) << (64 * index) for index in range(count))

    assert count == 3
    assert rebuilt == high_mask
    assert words.dtype.str == "<u8"


def test_all_f64_payloads_have_exact_factor_references(
    lowering_input: EagerLoweringInputV1,
) -> None:
    factors = lowering_input.table("exact_factors")
    factor_count = factors.row_count
    exact_ir_ids = factors.column("exact_ir_id")
    sources = factors.column("exact_source")

    assert factor_count > 0
    assert len(exact_ir_ids) == factor_count
    assert set(int(value) for value in sources) <= {
        FACTOR_EXACT_SOURCE_BINARY64,
        FACTOR_EXACT_SOURCE_CANONICAL_IR,
    }
    for row in range(factor_count):
        payload = json.loads(
            lowering_input.canonical_ir_catalog[int(exact_ir_ids[row])]
        )
        real = float(factors.column("real")[row])
        imaginary = float(factors.column("imaginary")[row])
        if int(sources[row]) == FACTOR_EXACT_SOURCE_BINARY64:
            assert payload == {
                "abi": "pyamplicol-exact-factor-v1",
                "imaginary_bits": _f64_bits(imaginary),
                "kind": "complex-ieee754-binary64",
                "real_bits": _f64_bits(real),
            }
        else:
            assert payload["kind"] == "canonical-source-ir"
            fallback = payload["f64_fallback"]
            assert fallback["real_bits"] == _f64_bits(real)
            assert fallback["imaginary_bits"] == _f64_bits(imaginary)

    for table in lowering_input.tables:
        for column in table.columns:
            if column.name.endswith("factor_id"):
                assert (
                    column.values.size == 0 or int(column.values.max()) < factor_count
                )

    f64_columns = {
        (table.name, column.name)
        for table in lowering_input.tables
        for column in table.columns
        if column.values.dtype.str == "<f8"
    }
    assert f64_columns == {
        ("exact_factors", "real"),
        ("exact_factors", "imaginary"),
        ("model_parameters", "default_value"),
    }
    parameter_table = lowering_input.table("model_parameters")
    for row, factor_id in enumerate(parameter_table.column("default_factor_id")):
        assert (
            parameter_table.column("default_value")[row]
            == factors.column("real")[int(factor_id)]
        )
        assert factors.column("imaginary")[int(factor_id)] == 0.0


def test_runtime_couplings_use_canonical_parameter_ir(
    lowering_input: EagerLoweringInputV1,
) -> None:
    couplings = lowering_input.table("couplings")
    factors = lowering_input.table("exact_factors")
    found = False
    for factor_id in couplings.column("constant_factor_id"):
        row = int(factor_id)
        if int(factors.column("exact_source")[row]) != (
            FACTOR_EXACT_SOURCE_CANONICAL_IR
        ):
            continue
        payload = json.loads(
            lowering_input.canonical_ir_catalog[int(factors.column("exact_ir_id")[row])]
        )
        source = payload["source"]
        assert source["kind"] == "model-parameter-components"
        assert any(
            component["kind"] == "model-parameter" for component in source["components"]
        )
        found = True
    assert found, "the electroweak fixture must exercise parameter-backed couplings"


def test_contraction_and_crossing_factors_retain_source_ir_ownership(
    lowering_input: EagerLoweringInputV1,
) -> None:
    factors = lowering_input.table("exact_factors")
    coefficients = lowering_input.table("contraction_coefficients")
    for row, factor_id in enumerate(coefficients.column("factor_id")):
        assert (
            factors.column("source_ir_id")[int(factor_id)]
            == (coefficients.column("contraction_ir_id")[row])
        )

    sources = lowering_input.table("sources")
    for row, factor_id in enumerate(sources.column("crossing_factor_id")):
        assert (
            factors.column("source_ir_id")[int(factor_id)]
            == (sources.column("crossing_ir_id")[row])
        )


def test_binary64_catalog_distinguishes_signed_zero() -> None:
    case = _case("g g > g g", prepared_closure=False)
    interactions = list(case.dag.interactions)
    interactions[0] = replace(interactions[0], color_weight=(0.0, 0.0))
    interactions[1] = replace(interactions[1], color_weight=(-0.0, 0.0))
    dag = replace(case.dag, interactions=tuple(interactions))
    result = build_eager_lowering_input_v1(
        dag=dag,
        model=case.model,
        resolver=_resolver_for(dag, case.model),
    )
    table = result.table("interactions")
    positive = int(table.column("color_factor_id")[0])
    negative = int(table.column("color_factor_id")[1])
    factors = result.table("exact_factors")

    assert positive != negative
    assert _f64_bits(float(factors.column("real")[positive])) == "0000000000000000"
    assert _f64_bits(float(factors.column("real")[negative])) == "8000000000000000"


def test_recurrence_proofs_and_structural_zeros_are_retained() -> None:
    case = _case("g g > g g", recurrence=True, prepared_closure=False)
    result = build_eager_lowering_input_v1(
        dag=case.dag,
        model=case.model,
        resolver=case.resolver,
    )
    proof = case.dag.helicity_recurrence
    materialization = case.dag.helicity_materialization
    assert proof is not None
    assert materialization is not None

    assert result.table("helicity_domains").row_count == len(proof.selector_domains)
    assert result.table("helicity_recurrence_classes").row_count == len(
        proof.recurrence_classes
    )
    assert result.table("helicity_materialized_source_routes").row_count == len(
        materialization.source_routes
    )
    assert result.table("helicity_materialized_amplitude_routes").row_count == len(
        materialization.amplitude_routes
    )
    assert result.table("helicity_structural_zero_domains").row_count == len(
        proof.structural_zero_selector_domain_ids
    )
    assert result.table("interaction_groups").row_count > 0
    assert result.table("interaction_group_members").row_count == len(
        case.dag.interactions
    )


def test_mixed_zero_group_selects_a_live_eager_representative() -> None:
    case = _case("g g > g g", prepared_closure=False)
    grouped: dict[int, list[object]] = {}
    for interaction in case.dag.interactions:
        assert interaction.evaluation_group_id is not None
        grouped.setdefault(
            int(interaction.evaluation_group_id),
            [],
        ).append(interaction)
    members = next(value for value in grouped.values() if len(value) > 1)
    zero_member = members[0]
    live_member = members[1]
    dag = replace(
        case.dag,
        interactions=tuple(
            replace(interaction, evaluation_factor=(0.0, 0.0))
            if interaction.id == zero_member.id
            else interaction
            for interaction in case.dag.interactions
        ),
    )

    result = build_eager_lowering_input_v1(
        dag=dag,
        model=case.model,
        resolver=_resolver_for(dag, case.model),
    )
    interaction_table = result.table("interactions")
    compact_group_id = int(
        interaction_table.column("evaluation_group_id")[zero_member.id]
    )
    group_table = result.table("interaction_groups")
    member_table = result.table("interaction_group_members")
    start = int(group_table.column("member_start")[compact_group_id])
    count = int(group_table.column("member_count")[compact_group_id])
    ordered_members = tuple(
        int(value)
        for value in member_table.column("interaction_id")[
            start : start + count
        ]
    )

    assert int(
        group_table.column("representative_interaction_id")[
            compact_group_id
        ]
    ) == live_member.id
    assert ordered_members[0] == live_member.id
    assert ordered_members[-1] == zero_member.id


@pytest.mark.parametrize("color_accuracy", ("nlc", "full"))
def test_contracted_color_reductions_are_columnar(color_accuracy: str) -> None:
    model = BuiltinSMModel()
    dag = compile_generic_dag(
        build_process_ir("g g > g g", color_accuracy=color_accuracy),
        model=model,
    )
    result = build_eager_lowering_input_v1(
        dag=dag,
        model=model,
        resolver=_resolver_for(dag, model),
    )
    metadata = result.table("color_contraction_metadata")
    entries = result.table("color_contraction_entries")
    reductions = result.table("reduction_members")
    _group_ids, descriptors = _amplitude_groups(dag, dag.amplitude_roots)
    contraction = build_color_contraction_plan(dag.color_plan, descriptors)

    assert contraction is not None
    assert contraction.repeated_block is not None
    logical_entries = tuple(contraction.iter_logical_entries())
    assert bool(metadata.column("present")[0])
    assert bool(metadata.column("supported")[0])
    assert entries.row_count == contraction.logical_entry_count > 0
    assert tuple(int(value) for value in entries.column("left_group_id")) == tuple(
        entry.left_group_id for entry in logical_entries
    )
    assert tuple(int(value) for value in entries.column("right_group_id")) == tuple(
        entry.right_group_id for entry in logical_entries
    )
    assert reductions.row_count > 0
    assert set(int(value) for value in reductions.column("color_selector_id")) == {0}
    assert result.table("color_selectors").row_count == 1
    factors = result.table("exact_factors")
    for factor_id, entry in zip(
        entries.column("weight_factor_id"), logical_entries, strict=True
    ):
        assert (
            int(factors.column("exact_source")[factor_id])
            == FACTOR_EXACT_SOURCE_CANONICAL_IR
        )
        ir_id = int(factors.column("exact_ir_id")[factor_id])
        source = json.loads(result.canonical_ir_catalog[ir_id])["source"]
        assert source == {
            "kind": "rational-complex", "value": entry.to_json_dict()["exact_weight"]
        }


def test_lc_replay_proof_ownership_is_retained() -> None:
    case = _case("g g > g g", prepared_closure=False)
    replay = build_lc_topology_replay_plan(case.dag.color_plan, case.model)
    dag = replace(case.dag, lc_topology_replay=replay)
    result = build_eager_lowering_input_v1(
        dag=dag,
        model=case.model,
        resolver=_resolver_for(dag, case.model),
    )
    metadata = result.table("lc_replay_metadata")
    members = result.table("lc_replay_members")

    assert bool(metadata.column("present")[0])
    assert result.table("lc_replay_partitions").row_count == len(replay.partitions)
    assert members.row_count == replay.replayed_sector_count
    assert result.table("lc_replay_residual_sectors").row_count == len(
        replay.residual_sector_ids
    )
    assert result.table("lc_replay_permutations").row_count == sum(
        len(permutation)
        for partition in replay.partitions
        for permutation in partition.label_permutations
    )


def test_generic_color_replay_retains_compact_physical_contraction() -> None:
    model = BuiltinSMModel()
    process = build_process_ir("g g > g g g", color_accuracy="full")
    color_plan = build_color_plan(process, color_accuracy="full")
    replay = build_color_topology_replay_certificate(color_plan, model)
    assert replay is not None
    dag = compile_generic_dag(
        process,
        model=model,
        color_plan=color_plan,
        selected_color_sector_ids=replay.materialized_sector_ids,
        online_evaluation_reuse=True,
        backward_live_planning=True,
    )
    dag = replace(
        dag,
        color_plan=color_plan,
        color_coverage="complete",
        selected_color_sector_ids=(),
        color_topology_replay=replay,
    )
    result = build_eager_lowering_input_v1(
        dag=dag,
        model=model,
        resolver=_resolver_for(dag, model),
    )

    replay_metadata = result.table("lc_replay_metadata")
    amplitude_metadata = result.table("color_replay_metadata")
    contraction_metadata = result.table("color_replay_contraction_metadata")
    component_groups = result.table("color_replay_contraction_component_groups")
    entries = result.table("color_replay_contraction_entries")

    assert bool(replay_metadata.column("present")[0])
    assert int(replay_metadata.column("replay_kind")[0]) == 2
    assert bool(amplitude_metadata.column("present")[0])
    assert (
        result.table("color_replay_physical_groups").row_count
        == int(amplitude_metadata.column("physical_group_count")[0])
    )
    assert bool(contraction_metadata.column("present")[0])
    assert bool(contraction_metadata.column("repeated")[0])
    component_count = int(contraction_metadata.column("component_count")[0])
    assert component_count > 1
    assert component_groups.row_count == int(
        contraction_metadata.column("group_count")[0]
    )
    assert entries.row_count > 0
    assert entries.row_count * component_count > entries.row_count
    assert set(int(value) for value in entries.column("left_group_id")) == {
        (1 << 32) - 1
    }
    assert set(int(value) for value in entries.column("right_group_id")) == {
        (1 << 32) - 1
    }
    assert not bool(result.table("color_contraction_metadata").column("present")[0])
    factors = result.table("exact_factors")
    for factor_id in entries.column("weight_factor_id"):
        assert (
            int(factors.column("exact_source")[factor_id])
            == FACTOR_EXACT_SOURCE_CANONICAL_IR
        )
        ir_id = int(factors.column("exact_ir_id")[factor_id])
        source = json.loads(result.canonical_ir_catalog[ir_id])["source"]
        assert source["kind"] == "rational-complex"
        assert len(source["value"]) == 4


def test_exactness_audit_reports_the_remaining_resolver_blocker(
    lowering_input: EagerLoweringInputV1,
) -> None:
    audit = " ".join(lowering_input.semantic_limitations)
    assert "exact-IR reference" in audit
    assert "intrinsically f64" in audit
    assert "prepared-kernel exact expressions" in audit.lower()
    assert "remaining provenance blocker" in audit
