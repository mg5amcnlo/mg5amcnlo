# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from fractions import Fraction
from itertools import permutations
from math import isclose, nextafter

import pytest

import pyamplicol.color as color
import pyamplicol.color.contraction_factors as contraction_factors
import pyamplicol.generation.artifact_writer as artifact_writer
from pyamplicol.color import (
    ColorContractionEntry,
    ColorContractionTemplateEntry,
    ColorGroupDescriptor,
    FactorizedColorContractionBlock,
    RepeatedColorContractionBlock,
    build_color_contraction_plan,
    build_color_plan,
    color_contraction_factor,
    color_contraction_factors,
    exact_color_contraction_factor,
    exact_color_contraction_factors,
)
from pyamplicol.color.contraction import _build_walsh_color_contraction_block
from pyamplicol.color.contraction_factors import (
    _pure_adjoint_full_factor_by_relative_permutation,
    _pure_adjoint_full_factor_uncached,
    _relative_adjoint_permutation,
)
from pyamplicol.generation.artifact_writer import _color_contraction
from pyamplicol.generation.dag_color import ColorEngine
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.dag_ordering import _sector_intermediate_order_words
from pyamplicol.models.builtin.model import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir


def test_color_one_open_line_factors_match_reference_convention() -> None:
    plan = build_color_plan(
        build_process_ir("d d~ > z g", color_accuracy="full"),
        color_accuracy="full",
    )
    sector = plan.sectors[0]
    assert color_contraction_factors(plan, sector, sector) == (9.0, 8.0, 8.0)


@pytest.mark.parametrize(
    "process",
    (
        "g g > g g",
        "d d~ > z g",
        "d d~ > u u~",
        "d d~ > u u~ s s~ g",
    ),
)
def test_exact_color_factors_match_binary64_reference_for_every_sector_pair(
    process: str,
) -> None:
    plan = build_color_plan(
        build_process_ir(process, color_accuracy="full"),
        color_accuracy="full",
    )

    for left in plan.sectors:
        for right in plan.sectors:
            exact = exact_color_contraction_factors(plan, left, right)
            binary64 = color_contraction_factors(plan, left, right)
            assert all(isinstance(value, Fraction) for value in exact)
            assert tuple(float(value) for value in exact) == binary64
            for accuracy, expected in zip(
                ("lc", "nlc", "full"),
                exact,
                strict=True,
            ):
                assert (
                    exact_color_contraction_factor(
                        plan,
                        left,
                        right,
                        accuracy=accuracy,
                    )
                    == expected
                )


def test_pure_adjoint_full_factors_cache_relative_permutations_exactly() -> None:
    words = tuple(permutations((11, 13, 17, 19)))
    _pure_adjoint_full_factor_by_relative_permutation.cache_clear()

    for left in words:
        for right in words:
            relative = _relative_adjoint_permutation(left, right)
            assert relative is not None
            cached = _pure_adjoint_full_factor_by_relative_permutation(
                relative,
                len(left),
                20,
            )
            direct = _pure_adjoint_full_factor_uncached(
                left,
                right,
                len(left),
                20,
            )
            assert cached == direct

    info = _pure_adjoint_full_factor_by_relative_permutation.cache_info()
    assert info.misses == len(words)
    assert info.hits == len(words) * (len(words) - 1)


def test_pure_adjoint_exact_nlc_does_not_compute_full_color(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = build_color_plan(
        build_process_ir("g g > g g", color_accuracy="nlc"),
        color_accuracy="nlc",
    )

    def reject_full_color(*args: object, **kwargs: object) -> None:
        raise AssertionError("NLC requested an unnecessary full-colour trace")

    monkeypatch.setattr(
        contraction_factors,
        "_pure_adjoint_full_factor_exact_by_relative_permutation",
        reject_full_color,
    )
    monkeypatch.setattr(
        contraction_factors,
        "_pure_adjoint_full_factor_exact_uncached",
        reject_full_color,
    )
    for left in plan.sectors:
        for right in plan.sectors:
            assert exact_color_contraction_factor(
                plan, left, right, accuracy="nlc"
            ) == Fraction(color_contraction_factor(plan, left, right, accuracy="nlc"))


def test_color_plan_json_exposes_structural_open_line_roles() -> None:
    plan = build_color_plan(build_process_ir("d d~ > z g"))
    sector_record = plan.sectors[0]

    assert isinstance(sector_record.open_color_lines[0], color.LCOpenColorLine)
    assert not hasattr(sector_record, "quark_lines")
    assert not hasattr(color, "LCQuarkLine")

    payload = plan.to_json_dict()
    sector = payload["sectors"][0]
    line = sector["open_color_lines"][0]

    assert set(sector) == {
        "id",
        "kind",
        "open_color_lines",
        "trace_labels",
        "singlet_labels",
        "word_labels",
        "coloured_label_groups",
        "line_label_groups",
        "color_words",
        "admissible_traversal_words",
    }
    assert "idenso_required" not in payload

    assert "quark_lines" not in sector
    assert line == {
        "fundamental_label": 2,
        "antifundamental_label": 1,
        "adjoint_labels": [4],
        "singlet_labels": [],
        "line_labels": [2, 4, 1],
    }
    assert not {"quark_label", "antiquark_label", "gluon_labels"} & line.keys()


def test_three_open_lines_keep_distinct_fixed_sink_traversals() -> None:
    process = build_process_ir("d d~ > u u~ s s~")
    plan = build_color_plan(process)
    sector = plan.sectors[0]

    assert sector.color_words == ((2, 1, 3, 4, 5, 6),)
    assert _sector_intermediate_order_words(sector, process=process) == (
        (2, 1, 3, 4, 5, 6),
        (3, 4, 2, 1, 5, 6),
    )


def test_three_open_line_nlc_keeps_exact_qualified_coefficients() -> None:
    plan = build_color_plan(
        build_process_ir(
            "d d~ > u u~ s s~ g",
            color_accuracy="full",
        ),
        color_accuracy="full",
    )
    sectors = {sector.word_labels: sector for sector in plan.sectors}
    reference = sectors[(2, 1, 3, 4, 5, 7, 6)]

    assert color_contraction_factors(plan, reference, reference) == (
        81.0,
        72.0,
        72.0,
    )
    assert color_contraction_factors(
        plan,
        reference,
        sectors[(2, 1, 3, 6, 5, 7, 4)],
    ) == (0.0, -24.0, -24.0)
    assert color_contraction_factors(
        plan,
        reference,
        sectors[(2, 4, 3, 6, 5, 7, 1)],
    ) == (0.0, 8.0, 8.0)


def test_nlc_one_open_line_recycles_runtime_route_safe_orderings() -> None:
    model = BuiltinSMModel()
    process = build_process_ir("g g > t t~ g", color_accuracy="nlc")
    plan = build_color_plan(process, color_accuracy="nlc")
    engine = ColorEngine(plan, model)
    dag = compile_generic_dag(process, model=model)

    assert plan.sector_count == 6
    assert engine.shared_lc_orderings is True
    assert len(dag.currents) == 250
    assert len(dag.interactions) == 624
    # Reversed ordered-label routes stay distinct even when their color
    # ordering is shared; only route-safe kernel evaluations may be recycled.
    assert dag.interaction_evaluation_count == 420
    assert len(dag.amplitude_roots) == 192


def test_nlc_multiple_open_lines_keep_pairing_identity_sector_local() -> None:
    model = BuiltinSMModel()
    process = build_process_ir("d d~ > u u~ s s~", color_accuracy="nlc")
    plan = build_color_plan(process, color_accuracy="nlc")
    engine = ColorEngine(plan, model)

    assert plan.sector_count > 1
    assert engine.shared_lc_orderings is False


def test_color_contraction_rejects_inconsistent_helicity_weights() -> None:
    plan = build_color_plan(
        build_process_ir("d d~ > z g", color_accuracy="full"),
        color_accuracy="full",
    )
    sector = plan.sectors[0]
    common = {
        "helicity_key": ("h:-1,+1,+0,+1",),
        "sector_id": sector.id,
        "word": tuple(sector.word_labels or sector.color_words[0]),
    }
    groups = (
        ColorGroupDescriptor(group_id=0, helicity_weight=1.0, **common),
        ColorGroupDescriptor(group_id=1, helicity_weight=0.5, **common),
    )

    with pytest.raises(ValueError, match="inconsistent helicity weights"):
        build_color_contraction_plan(plan, groups)


def test_color_contraction_compacts_identical_helicity_components() -> None:
    plan = build_color_plan(
        build_process_ir("g g > g g", color_accuracy="full"),
        color_accuracy="full",
    )
    first, second = plan.sectors[:2]
    first_word = tuple(first.word_labels or first.color_words[0])
    second_word = tuple(second.word_labels or second.color_words[0])
    groups = (
        ColorGroupDescriptor(
            group_id=10,
            helicity_key=("helicity:a",),
            sector_id=second.id,
            word=second_word,
            helicity_weight=1.0,
        ),
        ColorGroupDescriptor(
            group_id=20,
            helicity_key=("helicity:b",),
            sector_id=first.id,
            word=first_word,
            helicity_weight=1.0,
        ),
        ColorGroupDescriptor(
            group_id=30,
            helicity_key=("helicity:a",),
            sector_id=first.id,
            word=first_word,
            helicity_weight=1.0,
        ),
        ColorGroupDescriptor(
            group_id=40,
            helicity_key=("helicity:b",),
            sector_id=second.id,
            word=second_word,
            helicity_weight=1.0,
        ),
    )

    contraction = build_color_contraction_plan(plan, groups)

    assert contraction is not None
    assert contraction.entries == ()
    block = contraction.repeated_block
    assert block is not None
    assert block.component_count == 2
    assert block.component_group_ids == (30, 20, 10, 40)
    logical = tuple(contraction.iter_logical_entries())
    assert len(logical) == contraction.logical_entry_count
    assert tuple(
        (entry.left_group_id, entry.right_group_id) for entry in logical
    ) == tuple(
        (
            block.component_group_ids[
                entry.left_group_index * block.component_count + component
            ],
            block.component_group_ids[
                entry.right_group_index * block.component_count + component
            ],
        )
        for component in range(block.component_count)
        for entry in block.entries
    )
    payload = contraction.to_json_dict()
    assert payload["entry_count"] == 0
    assert payload["logical_entry_count"] == len(logical)
    assert payload["entries"] == []
    assert payload["repeated_block"] == block.to_json_dict()
    assert _color_contraction(payload)["repeated_block"] == block.to_json_dict()

    expanded = tuple(
        entry
        for helicity_key in (("helicity:a",), ("helicity:b",))
        for entry in (
            build_color_contraction_plan(
                plan,
                tuple(
                    descriptor
                    for descriptor in groups
                    if descriptor.helicity_key == helicity_key
                ),
            )
            or pytest.fail("single-component contraction is absent")
        ).entries
    )
    amplitudes = {
        10: complex(0.5, -1.0),
        20: complex(1.5, 0.25),
        30: complex(-0.75, 2.0),
        40: complex(0.125, -0.5),
    }

    def reduce(entries: tuple[ColorContractionEntry, ...]) -> float:
        total = 0.0
        for entry in entries:
            left = amplitudes[entry.left_group_id]
            right = amplitudes[entry.right_group_id]
            product = left * right.conjugate()
            total += entry.symmetry_factor * (
                entry.weight_re * product.real - entry.weight_im * product.imag
            )
        return total

    assert isclose(reduce(logical), reduce(expanded), rel_tol=1.0e-15, abs_tol=1.0e-15)


def test_full_color_permutation_orbit_emits_klein_four_walsh_plan() -> None:
    plan = build_color_plan(
        build_process_ir("d d~ > z g g g g", color_accuracy="full"),
        color_accuracy="full",
    )
    groups = tuple(
        ColorGroupDescriptor(
            group_id=component_index * len(plan.sectors) + sector_index,
            helicity_key=(f"helicity:{component_index}",),
            sector_id=sector.id,
            word=tuple(sector.word_labels or sector.color_words[0]),
            helicity_weight=1.0,
        )
        for component_index in range(2)
        for sector_index, sector in enumerate(plan.sectors)
    )

    contraction = build_color_contraction_plan(plan, groups)

    assert contraction is not None
    block = contraction.repeated_block
    assert block is not None
    factorized = block.factorized_block
    assert factorized is not None
    assert factorized.kind == "klein-four-walsh"
    assert factorized.rank is None
    assert len(factorized.cosets) == len(plan.sectors) // 4
    assert sorted(index for coset in factorized.cosets for index in coset) == list(
        range(len(plan.sectors))
    )
    payload = contraction.to_json_dict()
    assert "rank" not in payload["repeated_block"]["factorized_block"]
    assert payload["repeated_block"] == block.to_json_dict()
    assert _color_contraction(payload)["repeated_block"] == block.to_json_dict()
    assert artifact_writer._runtime_schema_uses_walsh_color_contraction(
        {"amplitude_stage": {"color_contraction": payload}}
    )


def _elementary_abelian_orbit_words(rank: int) -> tuple[tuple[int, ...], ...]:
    labels = tuple(range(2 * rank))
    return tuple(
        tuple(label ^ 1 if mask & (1 << (label // 2)) else label for label in labels)
        for mask in range(1 << rank)
    )


def _diagonal_color_entries(
    count: int,
    *,
    exceptional_weight: float | None = None,
) -> tuple[ColorContractionTemplateEntry, ...]:
    return tuple(
        ColorContractionTemplateEntry(
            left_group_index=index,
            right_group_index=index,
            weight_re=(
                exceptional_weight
                if index == 0 and exceptional_weight is not None
                else 1.0
            ),
        )
        for index in range(count)
    )


def test_elementary_abelian_walsh_plan_emits_measured_rank_three() -> None:
    rank = 3
    words = _elementary_abelian_orbit_words(rank)

    factorized = _build_walsh_color_contraction_block(
        words,
        _diagonal_color_entries(len(words)),
    )

    assert factorized is not None
    assert factorized.kind == "elementary-abelian-walsh"
    assert factorized.rank == rank
    assert factorized.cosets == (tuple(range(1 << rank)),)
    assert factorized.to_json_dict() == {
        "kind": "elementary-abelian-walsh",
        "rank": rank,
        "cosets": [list(range(1 << rank))],
    }
    contraction_payload = {
        "supported": True,
        "reason": None,
        "group_count": 2 * len(words),
        "includes_color_factor": True,
        "entries": [],
        "repeated_block": {
            "component_count": 2,
            "component_group_ids": list(range(2 * len(words))),
            "entries": [],
            "factorized_block": factorized.to_json_dict(),
        },
    }
    assert (
        _color_contraction(contraction_payload)["repeated_block"]["factorized_block"]
        == factorized.to_json_dict()
    )
    assert artifact_writer._runtime_schema_walsh_color_contraction_capabilities(
        {"amplitude_stage": {"color_contraction": contraction_payload}}
    ) == frozenset({artifact_writer.COMPILED_COLOR_CONTRACTION_WALSH_C2K_CAPABILITY})


def test_elementary_abelian_walsh_plan_caps_wider_orbit_at_rank_three() -> None:
    words = _elementary_abelian_orbit_words(4)

    factorized = _build_walsh_color_contraction_block(
        words,
        _diagonal_color_entries(len(words)),
    )

    assert factorized is not None
    assert factorized.kind == "elementary-abelian-walsh"
    assert factorized.rank == 3
    assert factorized.cosets == (
        tuple(range(8)),
        tuple(range(8, 16)),
    )


def test_rank_three_walsh_recognizes_multi_coset_xor_matrix_exactly() -> None:
    rank = 3
    subgroup_order = 1 << rank
    seeds = (
        tuple(range(2 * rank)),
        (0, 2, 1, 3, 4, 5),
    )
    words = tuple(
        tuple(label ^ 1 if mask & (1 << (label // 2)) else label for label in seed)
        for seed in seeds
        for mask in range(subgroup_order)
    )

    def matrix_weight(left: int, right: int) -> float:
        left_coset, left_subgroup = divmod(left, subgroup_order)
        right_coset, right_subgroup = divmod(right, subgroup_order)
        coset_pair = tuple(sorted((left_coset, right_coset)))
        pair_offset = {
            (0, 0): 8,
            (0, 1): 24,
            (1, 1): 40,
        }[coset_pair]
        return (pair_offset + (left_subgroup ^ right_subgroup) + 1) / 8.0

    entries = tuple(
        ColorContractionTemplateEntry(
            left_group_index=left,
            right_group_index=right,
            weight_re=matrix_weight(left, right),
            symmetry_factor=1.0 if left == right else 2.0,
        )
        for left in range(len(words))
        for right in range(left, len(words))
    )

    factorized = _build_walsh_color_contraction_block(words, entries)

    assert factorized is not None
    assert factorized.kind == "elementary-abelian-walsh"
    assert factorized.rank == rank
    assert factorized.cosets == (
        tuple(range(subgroup_order)),
        tuple(range(subgroup_order, 2 * subgroup_order)),
    )
    expected_wire = {
        "kind": "elementary-abelian-walsh",
        "rank": rank,
        "cosets": [
            list(range(subgroup_order)),
            list(range(subgroup_order, 2 * subgroup_order)),
        ],
    }
    assert factorized.to_json_dict() == expected_wire

    component_count = 3
    contraction_payload = {
        "supported": True,
        "reason": None,
        "group_count": component_count * len(words),
        "includes_color_factor": True,
        "entries": [],
        "repeated_block": {
            "component_count": component_count,
            "component_group_ids": list(range(component_count * len(words))),
            "entries": [entry.to_json_dict() for entry in entries],
            "factorized_block": factorized.to_json_dict(),
        },
    }
    assert (
        _color_contraction(contraction_payload)["repeated_block"]["factorized_block"]
        == expected_wire
    )

    amplitudes = tuple(
        tuple(
            complex(
                (local_group + 1) * (component + 2) / 13.0,
                (local_group - 2 * component) / 17.0,
            )
            for component in range(component_count)
        )
        for local_group in range(len(words))
    )
    direct = sum(
        matrix_weight(left, right)
        * sum(
            (
                amplitudes[left][component] * amplitudes[right][component].conjugate()
            ).real
            for component in range(component_count)
        )
        for left in range(len(words))
        for right in range(len(words))
    )

    def walsh(values: list[complex] | list[float]) -> None:
        stride = 1
        while stride < len(values):
            for start in range(0, len(values), 2 * stride):
                for offset in range(stride):
                    left = values[start + offset]
                    right = values[start + stride + offset]
                    values[start + offset] = left + right
                    values[start + stride + offset] = left - right
            stride *= 2

    transformed = [
        [
            [
                amplitudes[coset * subgroup_order + subgroup][component]
                for subgroup in range(subgroup_order)
            ]
            for component in range(component_count)
        ]
        for coset in range(len(seeds))
    ]
    for coset_components in transformed:
        for component_values in coset_components:
            walsh(component_values)

    factorized_total = 0.0
    for left_coset in range(len(seeds)):
        for right_coset in range(left_coset, len(seeds)):
            weights = [
                matrix_weight(
                    left_coset * subgroup_order,
                    right_coset * subgroup_order + subgroup,
                )
                for subgroup in range(subgroup_order)
            ]
            walsh(weights)
            symmetry_factor = 1.0 if left_coset == right_coset else 2.0
            for character, weight in enumerate(weights):
                product = sum(
                    (
                        transformed[left_coset][component][character]
                        * transformed[right_coset][component][character].conjugate()
                    ).real
                    for component in range(component_count)
                )
                factorized_total += symmetry_factor * weight * product / subgroup_order

    assert isclose(
        factorized_total,
        direct,
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    )


@pytest.mark.parametrize(
    ("kind", "rank", "cosets", "message"),
    [
        (
            "klein-four-walsh",
            2,
            ((0, 1, 2, 3),),
            "cannot declare rank",
        ),
        (
            "elementary-abelian-walsh",
            None,
            (tuple(range(8)),),
            "requires rank >= 3",
        ),
        (
            "elementary-abelian-walsh",
            2,
            ((0, 1, 2, 3),),
            "requires rank >= 3",
        ),
        (
            "elementary-abelian-walsh",
            3,
            ((0, 1, 2, 3),),
            "rank is inconsistent",
        ),
    ],
)
def test_factorized_walsh_contract_rejects_malformed_rank_metadata(
    kind: str,
    rank: int | None,
    cosets: tuple[tuple[int, ...], ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        FactorizedColorContractionBlock(kind=kind, rank=rank, cosets=cosets)


def test_factorized_walsh_contract_rejects_partial_local_partition() -> None:
    factorized = FactorizedColorContractionBlock(
        kind="elementary-abelian-walsh",
        rank=3,
        cosets=(tuple(range(8)),),
    )

    with pytest.raises(ValueError, match="do not partition local groups"):
        RepeatedColorContractionBlock(
            component_count=2,
            component_group_ids=tuple(range(18)),
            entries=(),
            factorized_block=factorized,
        )


def test_complete_six_label_orbit_emits_rank_three_walsh_plan() -> None:
    words = tuple(permutations(range(6)))

    factorized = _build_walsh_color_contraction_block(
        words,
        _diagonal_color_entries(len(words)),
    )

    assert factorized is not None
    assert factorized.kind == "elementary-abelian-walsh"
    assert factorized.rank == 3
    assert len(factorized.cosets) == len(words) // 8
    assert sorted(index for coset in factorized.cosets for index in coset) == list(
        range(len(words))
    )


def test_partial_or_asymmetric_rank_three_orbit_falls_back_safely() -> None:
    words = _elementary_abelian_orbit_words(3)

    partial = _build_walsh_color_contraction_block(
        words[:-1],
        _diagonal_color_entries(len(words) - 1),
    )
    asymmetric = _build_walsh_color_contraction_block(
        words,
        _diagonal_color_entries(len(words), exceptional_weight=2.0),
    )

    assert partial is None
    assert asymmetric is None


def test_nlc_permutation_orbit_emits_klein_four_walsh_plan() -> None:
    nlc_plan = build_color_plan(
        build_process_ir("g g > t t~ g g", color_accuracy="nlc"),
        color_accuracy="nlc",
    )
    nlc_groups = tuple(
        ColorGroupDescriptor(
            group_id=component_index * len(nlc_plan.sectors) + sector_index,
            helicity_key=(f"helicity:{component_index}",),
            sector_id=sector.id,
            word=tuple(sector.word_labels or sector.color_words[0]),
            helicity_weight=1.0,
        )
        for component_index in range(2)
        for sector_index, sector in enumerate(nlc_plan.sectors)
    )
    nlc_contraction = build_color_contraction_plan(nlc_plan, nlc_groups)
    assert nlc_contraction is not None
    assert nlc_contraction.repeated_block is not None
    factorized = nlc_contraction.repeated_block.factorized_block
    assert factorized is not None
    assert factorized.kind == "klein-four-walsh"
    assert len(factorized.cosets) == len(nlc_plan.sectors) // 4
    assert sorted(index for coset in factorized.cosets for index in coset) == list(
        range(len(nlc_plan.sectors))
    )
    assert artifact_writer._runtime_schema_uses_walsh_color_contraction(
        {"amplitude_stage": {"color_contraction": nlc_contraction.to_json_dict()}}
    )


def test_malformed_nlc_permutation_orbit_falls_back_safely() -> None:
    nlc_plan = build_color_plan(
        build_process_ir("g g > t t~ g g", color_accuracy="nlc"),
        color_accuracy="nlc",
    )
    duplicate_word = tuple(
        nlc_plan.sectors[0].word_labels or nlc_plan.sectors[0].color_words[0]
    )
    malformed_groups = tuple(
        ColorGroupDescriptor(
            group_id=component_index * len(nlc_plan.sectors) + sector_index,
            helicity_key=(f"helicity:{component_index}",),
            sector_id=sector.id,
            word=(
                duplicate_word
                if sector_index == 1
                else tuple(sector.word_labels or sector.color_words[0])
            ),
            helicity_weight=1.0,
        )
        for component_index in range(2)
        for sector_index, sector in enumerate(nlc_plan.sectors)
    )
    malformed_contraction = build_color_contraction_plan(nlc_plan, malformed_groups)
    assert malformed_contraction is not None
    assert malformed_contraction.repeated_block is not None
    assert malformed_contraction.repeated_block.factorized_block is None
    assert not artifact_writer._runtime_schema_uses_walsh_color_contraction(
        {"amplitude_stage": {"color_contraction": malformed_contraction.to_json_dict()}}
    )


def test_small_nlc_permutation_orbit_falls_back_safely() -> None:
    plan = build_color_plan(
        build_process_ir("g g > t t~ g", color_accuracy="nlc"),
        color_accuracy="nlc",
    )
    groups = tuple(
        ColorGroupDescriptor(
            group_id=component_index * len(plan.sectors) + sector_index,
            helicity_key=(f"helicity:{component_index}",),
            sector_id=sector.id,
            word=tuple(sector.word_labels or sector.color_words[0]),
            helicity_weight=1.0,
        )
        for component_index in range(2)
        for sector_index, sector in enumerate(plan.sectors)
    )

    contraction = build_color_contraction_plan(plan, groups)

    assert contraction is not None
    assert contraction.repeated_block is not None
    assert contraction.repeated_block.factorized_block is None


@pytest.mark.parametrize(("accuracy", "entry_count"), [("nlc", 384), ("full", 984)])
def test_two_open_line_orbit_discovers_exact_klein_four_action(
    accuracy: str,
    entry_count: int,
) -> None:
    plan = build_color_plan(
        build_process_ir("d d~ > t t~ g g g", color_accuracy=accuracy),
        color_accuracy=accuracy,
    )
    sectors = tuple(
        sector
        for sector in plan.sectors
        if (1 <= sector.id < 48 and sector.id % 2 == 1)
        or (48 <= sector.id < 96 and sector.id % 2 == 0)
    )
    groups = tuple(
        ColorGroupDescriptor(
            group_id=component_index * len(sectors) + sector_index,
            helicity_key=(f"helicity:{component_index}",),
            sector_id=sector.id,
            word=tuple(sector.word_labels or sector.color_words[0]),
            helicity_weight=1.0,
        )
        for component_index in range(2)
        for sector_index, sector in enumerate(sectors)
    )

    contraction = build_color_contraction_plan(plan, groups)

    assert len(sectors) == 48
    assert contraction is not None
    block = contraction.repeated_block
    assert block is not None
    assert len(block.entries) == entry_count
    factorized = block.factorized_block
    assert factorized is not None
    assert len(factorized.cosets) == len(sectors) // 4
    assert sorted(index for coset in factorized.cosets for index in coset) == list(
        range(len(sectors))
    )


def test_partial_two_open_line_orbit_falls_back_safely() -> None:
    plan = build_color_plan(
        build_process_ir("d d~ > t t~ g g g", color_accuracy="nlc"),
        color_accuracy="nlc",
    )
    sectors = tuple(
        sector
        for sector in plan.sectors
        if (1 <= sector.id < 48 and sector.id % 2 == 1)
        or (48 <= sector.id < 96 and sector.id % 2 == 0)
    )[:-1]
    groups = tuple(
        ColorGroupDescriptor(
            group_id=component_index * len(sectors) + sector_index,
            helicity_key=(f"helicity:{component_index}",),
            sector_id=sector.id,
            word=tuple(sector.word_labels or sector.color_words[0]),
            helicity_weight=1.0,
        )
        for component_index in range(2)
        for sector_index, sector in enumerate(sectors)
    )

    contraction = build_color_contraction_plan(plan, groups)

    assert contraction is not None
    assert contraction.repeated_block is not None
    assert contraction.repeated_block.factorized_block is None


def test_color_contraction_compaction_falls_back_for_nonidentical_components() -> None:
    plan = build_color_plan(
        build_process_ir("g g > g g", color_accuracy="full"),
        color_accuracy="full",
    )
    sector = plan.sectors[0]
    common = {
        "sector_id": sector.id,
        "word": tuple(sector.word_labels or sector.color_words[0]),
    }
    groups = (
        ColorGroupDescriptor(
            group_id=0,
            helicity_key=("helicity:a",),
            helicity_weight=1.0,
            **common,
        ),
        ColorGroupDescriptor(
            group_id=1,
            helicity_key=("helicity:b",),
            helicity_weight=nextafter(1.0, 2.0),
            **common,
        ),
    )

    contraction = build_color_contraction_plan(plan, groups)

    assert contraction is not None
    assert contraction.repeated_block is None
    assert contraction.logical_entry_count == len(contraction.entries) == 2


def test_color_contraction_public_names_have_no_legacy_aliases() -> None:
    assert callable(color.color_contraction_factor)
    assert callable(color.color_contraction_factors)
    assert callable(color.exact_color_contraction_factor)
    assert callable(color.exact_color_contraction_factors)
    assert not hasattr(color, "amplicol_color_factor")
    assert not hasattr(color, "amplicol_color_factors")
