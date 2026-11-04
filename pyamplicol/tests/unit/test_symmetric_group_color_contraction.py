# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib
import math
from collections import Counter
from fractions import Fraction
from itertools import permutations

import pytest

from pyamplicol.color import (
    ColorGroupDescriptor,
    build_color_plan,
    build_symmetric_group_color_contraction_plan,
    certify_symmetric_group_orbits,
    exact_color_contraction_factor,
    reconstruct_symmetric_group_dense_exact,
)
from pyamplicol.generation.recurrence_physics import (
    build_on_the_fly_color_contraction,
    on_the_fly_color_sector_owner_map,
)
from pyamplicol.generation.service import (
    _build_on_the_fly_contracted_color_payload_v1,
)
from pyamplicol.models.builtin.process_ir import build_process_ir


def _plan(process: str, *, accuracy: str = "full"):
    return build_color_plan(
        build_process_ir(process, color_accuracy=accuracy),
        color_accuracy=accuracy,
    )


def _owner_descriptors(color_plan):
    owners = on_the_fly_color_sector_owner_map(color_plan)
    owner_sector_ids = tuple(
        sector_id for sector_id, owner_id in enumerate(owners) if sector_id == owner_id
    )
    descriptors = tuple(
        ColorGroupDescriptor(
            group_id=group_id,
            helicity_key=(),
            sector_id=sector_id,
            word=color_plan.sectors[sector_id].color_words[0],
            helicity_weight=1.0,
        )
        for group_id, sector_id in enumerate(owner_sector_ids)
    )
    return owners, owner_sector_ids, descriptors


def _assert_dense_reconstruction(color_plan, block) -> None:
    dense = reconstruct_symmetric_group_dense_exact(block)
    sectors_by_id = {sector.id: sector for sector in color_plan.sectors}
    for left_index, left_sector_id in enumerate(block.local_sector_ids):
        for right_index in range(left_index, len(block.local_sector_ids)):
            expected = exact_color_contraction_factor(
                color_plan,
                sectors_by_id[left_sector_id],
                sectors_by_id[block.local_sector_ids[right_index]],
                accuracy=color_plan.color_accuracy,
            )
            assert dense.get((left_index, right_index), Fraction(0)) == expected


@pytest.mark.parametrize("accuracy", ("nlc", "full"))
def test_pure_trace_orbit_is_anchored_and_reconstructs_dense_metric(
    accuracy: str,
) -> None:
    color_plan = _plan("g g > g g", accuracy=accuracy)
    owners, owner_sector_ids, descriptors = _owner_descriptors(color_plan)

    partition = certify_symmetric_group_orbits(
        color_plan,
        owner_sector_ids,
        sector_owner_ids=owners,
    )
    contraction = build_symmetric_group_color_contraction_plan(
        color_plan,
        descriptors,
        sector_owner_ids=owners,
    )

    assert partition.fixed_adjoint_labels == (1,)
    assert partition.permuted_adjoint_labels == (2, 3, 4)
    assert partition.residual_sector_ids == ()
    assert len(partition.orbits) == 1
    assert partition.orbits[0].sector_ids == tuple(range(6))
    block = contraction.symmetric_group_block
    assert block is not None
    assert block.degree == 3
    assert block.channel_cosets == (tuple(range(6)),)
    assert block.local_sector_ids == tuple(range(6))
    assert len(block.kernel_entries) == 6
    assert block.residual_entries == ()
    assert block.hermiticity_check_mode == "vacuous"
    assert block.hermiticity_relative_indices == ()
    assert block.hermiticity_check_count == 0
    _assert_dense_reconstruction(color_plan, block)


def test_one_open_line_uses_every_adjoint_in_regular_orbit() -> None:
    color_plan = _plan("d d~ > z g g g")
    owners, owner_sector_ids, descriptors = _owner_descriptors(color_plan)

    contraction = build_symmetric_group_color_contraction_plan(
        color_plan,
        descriptors,
        sector_owner_ids=owners,
    )

    block = contraction.symmetric_group_block
    assert block is not None
    assert block.degree == 3
    assert block.channel_count == 1
    assert block.local_sector_ids == owner_sector_ids
    _assert_dense_reconstruction(color_plan, block)


def test_on_the_fly_builder_selects_symmetric_group_planner() -> None:
    color_plan = _plan("g g > g g")

    contraction, owner_by_sector, owner_sector_ids = build_on_the_fly_color_contraction(
        color_plan,
        contraction="symmetric-group-fft",
    )

    assert owner_by_sector == tuple(range(6))
    assert owner_sector_ids == tuple(range(6))
    assert contraction.symmetric_group_block is not None
    assert contraction.symmetric_group_block.degree == 3
    assert contraction.destination_by_group == tuple(range(6))


def test_on_the_fly_fft_summary_exposes_kernel_storage_and_provenance() -> None:
    color_plan = _plan("g g > g g")

    encoded = _build_on_the_fly_contracted_color_payload_v1(
        color_plan,
        contraction="symmetric-group-fft",
    )

    assert encoded.summary["storage"] == "convolution-kernels"
    assert encoded.summary["entry_count"] == 6
    assert encoded.summary["factorization"] == {
        "kind": "symmetric-group-fourier",
        "rank": 3,
        "coset_count": 1,
    }
    assert encoded.summary["fft_provenance"] == {
        "method": "symmetric-group-fourier",
        "degree": 3,
        "channel_count": 1,
        "covered_local_group_count": 6,
        "residual_group_count": 0,
        "residual_entry_count": 0,
        "raw_kernel_bytes": 96,
        "transformed_kernel_bytes": 48,
        "capability": "rusticol.color-contraction.symmetric-group-fft.v1",
    }


@pytest.mark.parametrize("accuracy", ("nlc", "full"))
def test_multi_open_line_channels_keep_zero_rows_and_reconstruct_dense_metric(
    accuracy: str,
) -> None:
    color_plan = _plan("d d~ > u u~ g g", accuracy=accuracy)
    owners, _, descriptors = _owner_descriptors(color_plan)

    contraction = build_symmetric_group_color_contraction_plan(
        color_plan,
        descriptors,
        sector_owner_ids=owners,
    )

    block = contraction.symmetric_group_block
    assert block is not None
    assert block.degree == 2
    assert block.channel_count == 6
    assert block.group_order == 2
    assert len(block.kernel_entries) == 6 * 7 // 2 * 2
    assert any(weight == 0 for weight in block.kernel_exact_weights)
    assert block.residual_local_group_indices == ()
    assert block.hermiticity_check_mode == "full"
    assert block.hermiticity_relative_indices == (0, 1)
    assert block.hermiticity_check_count == 6 * 5 // 2 * 2
    _assert_dense_reconstruction(color_plan, block)


def test_identical_open_lines_keep_endpoint_pairing_channels_separate() -> None:
    color_plan = _plan("d d~ > d d~ g g")
    owners, owner_sector_ids, descriptors = _owner_descriptors(color_plan)

    partition = certify_symmetric_group_orbits(
        color_plan,
        owner_sector_ids,
        sector_owner_ids=owners,
    )
    contraction = build_symmetric_group_color_contraction_plan(
        color_plan,
        descriptors,
        sector_owner_ids=owners,
    )

    block = contraction.symmetric_group_block
    assert block is not None
    assert block.degree == 2
    assert block.channel_count == 6
    sectors_by_id = {sector.id: sector for sector in color_plan.sectors}
    endpoint_pairings = [
        tuple(
            sorted(
                (
                    line.fundamental_label,
                    line.antifundamental_label,
                )
                for line in sectors_by_id[orbit.sector_ids[0]].open_color_lines
            )
        )
        for orbit in partition.orbits
    ]
    assert sorted(Counter(endpoint_pairings).values()) == [3, 3]
    _assert_dense_reconstruction(color_plan, block)


def test_whole_open_line_replicas_collapse_to_owner_destination_projection() -> None:
    color_plan = _plan("d d~ > u u~ g g")
    owners = on_the_fly_color_sector_owner_map(color_plan)
    component_count = 2
    descriptors = tuple(
        ColorGroupDescriptor(
            group_id=sector.id * component_count + component,
            helicity_key=(component,),
            sector_id=sector.id,
            word=sector.color_words[0],
            helicity_weight=1.0,
        )
        for sector in color_plan.sectors
        for component in range(component_count)
    )

    contraction = build_symmetric_group_color_contraction_plan(
        color_plan,
        descriptors,
        sector_owner_ids=owners,
    )

    block = contraction.symmetric_group_block
    assert block is not None
    # There are 24 traversal sectors, but only 12 canonical tensor owners.
    assert len(descriptors) == 48
    assert block.channel_count == 6
    assert block.local_group_count == 12
    assert contraction.group_count == 24
    assert block.component_group_ids == tuple(range(24))
    assert contraction.destination_by_group == tuple(
        sector_id * component_count + component
        for sector_id in block.local_sector_ids
        for component in range(component_count)
    )
    assert all(owners[sector_id] == sector_id for sector_id in block.local_sector_ids)
    _assert_dense_reconstruction(color_plan, block)

    reversed_contraction = build_symmetric_group_color_contraction_plan(
        color_plan,
        tuple(reversed(descriptors)),
        sector_owner_ids=owners,
    )
    assert reversed_contraction.destination_by_group == (
        contraction.destination_by_group
    )
    assert (
        reversed_contraction.symmetric_group_block.component_group_ids
        == block.component_group_ids
    )


def test_four_adjoint_two_line_plan_is_ten_coupled_s_four_owner_channels() -> None:
    color_plan = _plan("d d~ > u u~ g g g g")
    owners = on_the_fly_color_sector_owner_map(color_plan)
    descriptors = tuple(
        ColorGroupDescriptor(
            group_id=sector.id,
            helicity_key=(),
            sector_id=sector.id,
            word=sector.color_words[0],
            helicity_weight=1.0,
        )
        for sector in color_plan.sectors
    )

    contraction = build_symmetric_group_color_contraction_plan(
        color_plan,
        descriptors,
        sector_owner_ids=owners,
    )

    block = contraction.symmetric_group_block
    assert block is not None
    assert color_plan.sector_count == 480
    assert len(set(owners)) == 240
    assert block.degree == 4
    assert block.group_order == 24
    assert block.channel_count == 10
    assert block.local_group_count == 240
    assert block.channel_cosets == tuple(
        tuple(range(channel * 24, (channel + 1) * 24)) for channel in range(10)
    )
    assert len(block.kernel_entries) == 10 * 11 // 2 * 24
    assert block.residual_local_group_indices == ()
    assert block.hermiticity_check_mode == "full"
    assert block.hermiticity_relative_indices == tuple(range(24))
    assert block.hermiticity_check_count == 10 * 9 // 2 * 24
    assert all(owners[sector_id] == sector_id for sector_id in block.local_sector_ids)
    assert contraction.destination_by_group == block.local_sector_ids

    dense = reconstruct_symmetric_group_dense_exact(block)
    sectors_by_id = {sector.id: sector for sector in color_plan.sectors}
    for left_index, right_index in (
        (0, 0),
        (0, 23),
        (1, 23),
        (0, 24),
        (23, 239),
        (100, 200),
        (239, 239),
    ):
        assert dense[(left_index, right_index)] == exact_color_contraction_factor(
            color_plan,
            sectors_by_id[block.local_sector_ids[left_index]],
            sectors_by_id[block.local_sector_ids[right_index]],
            accuracy="full",
        )


@pytest.mark.parametrize("accuracy", ("nlc", "full"))
def test_incomplete_channel_is_an_exact_direct_residual_and_cross_metric(
    accuracy: str,
) -> None:
    color_plan = _plan("d d~ > u u~ g g", accuracy=accuracy)
    owners, owner_sector_ids, _ = _owner_descriptors(color_plan)
    complete_channel = (owner_sector_ids[0], owner_sector_ids[3])
    incomplete_member = owner_sector_ids[1]
    selected = (*complete_channel, incomplete_member)
    descriptors = tuple(
        ColorGroupDescriptor(
            group_id=index,
            helicity_key=(),
            sector_id=sector_id,
            word=color_plan.sectors[sector_id].color_words[0],
            helicity_weight=1.0,
        )
        for index, sector_id in enumerate(selected)
    )

    contraction = build_symmetric_group_color_contraction_plan(
        color_plan,
        descriptors,
        sector_owner_ids=owners,
    )

    block = contraction.symmetric_group_block
    assert block is not None
    assert block.channel_count == 1
    assert block.residual_local_group_indices == (2,)
    assert block.local_sector_ids[-1] == incomplete_member
    assert block.residual_entries
    assert tuple(
        (entry.left_group_index, entry.right_group_index)
        for entry in block.residual_entries
    ) == ((0, 2), (1, 2), (2, 2))
    assert block.residual_exact_weights[:2] == (Fraction(0), Fraction(0))
    assert all(
        2 in (entry.left_group_index, entry.right_group_index)
        for entry in block.residual_entries
    )
    _assert_dense_reconstruction(color_plan, block)


def test_nontrivial_degree_without_a_complete_orbit_is_unsupported() -> None:
    color_plan = _plan("d d~ > u u~ g g")
    owners, owner_sector_ids, _ = _owner_descriptors(color_plan)
    sector_id = owner_sector_ids[0]
    descriptors = (
        ColorGroupDescriptor(
            group_id=0,
            helicity_key=(),
            sector_id=sector_id,
            word=color_plan.sectors[sector_id].color_words[0],
            helicity_weight=1.0,
        ),
    )

    partition = certify_symmetric_group_orbits(
        color_plan,
        (sector_id,),
        sector_owner_ids=owners,
    )
    contraction = build_symmetric_group_color_contraction_plan(
        color_plan,
        descriptors,
        sector_owner_ids=owners,
    )

    assert partition.degree == 2
    assert partition.orbits == ()
    assert partition.residual_sector_ids == (sector_id,)
    assert not contraction.supported
    assert contraction.symmetric_group_block is None
    assert contraction.reason == (
        "symmetric-group FFT found no certified orbit at nontrivial degree 2"
    )


@pytest.mark.parametrize("accuracy", ("nlc", "full"))
def test_three_open_lines_with_two_adjoints_reconstruct_exact_s_two_interference(
    accuracy: str,
) -> None:
    color_plan = _plan("d d~ > u u~ s s~ g g", accuracy=accuracy)
    owners, _, descriptors = _owner_descriptors(color_plan)

    contraction = build_symmetric_group_color_contraction_plan(
        color_plan,
        descriptors,
        sector_owner_ids=owners,
    )

    block = contraction.symmetric_group_block
    assert block is not None
    assert block.degree == 2
    assert block.group_order == 2
    assert block.channel_count > 1
    sectors_by_id = {sector.id: sector for sector in color_plan.sectors}
    assert {
        len(sectors_by_id[block.local_sector_ids[coset[0]]].open_color_lines)
        for coset in block.channel_cosets
    } == {3}
    dense = reconstruct_symmetric_group_dense_exact(block)
    assert any(
        dense.get((left_coset[0], right_coset[0]), Fraction(0)) != 0
        for left_channel, left_coset in enumerate(block.channel_cosets)
        for right_coset in block.channel_cosets[left_channel + 1 :]
    )
    _assert_dense_reconstruction(color_plan, block)


def test_s_zero_and_s_one_use_owner_collapsed_direct_residual() -> None:
    for process in ("d d~ > u u~", "d d~ > u u~ s s~ g"):
        color_plan = _plan(process)
        owners = on_the_fly_color_sector_owner_map(color_plan)
        descriptors = tuple(
            ColorGroupDescriptor(
                group_id=sector.id,
                helicity_key=(),
                sector_id=sector.id,
                word=sector.color_words[0],
                helicity_weight=1.0,
            )
            for sector in color_plan.sectors
        )

        contraction = build_symmetric_group_color_contraction_plan(
            color_plan,
            descriptors,
            sector_owner_ids=owners,
        )

        assert contraction.supported
        assert contraction.symmetric_group_block is None
        assert contraction.entries
        assert contraction.group_count == len(set(owners))
        assert contraction.destination_by_group is not None
        assert len(contraction.destination_by_group) == contraction.group_count


def test_owner_map_must_preserve_the_canonical_color_tensor() -> None:
    color_plan = _plan("g g > g g")
    _, owner_sector_ids, _ = _owner_descriptors(color_plan)
    invalid_owners = (0, 0, 2, 3, 4, 5)

    with pytest.raises(ValueError, match="canonical color tensor"):
        certify_symmetric_group_orbits(
            color_plan,
            owner_sector_ids,
            sector_owner_ids=invalid_owners,
        )


def test_alias_only_descriptors_are_projected_to_authenticated_owners() -> None:
    color_plan = _plan("d d~ > u u~ g g")
    owners = on_the_fly_color_sector_owner_map(color_plan)
    aliases_by_owner: dict[int, list[int]] = {}
    for sector_id, owner_id in enumerate(owners):
        aliases_by_owner.setdefault(owner_id, []).append(sector_id)
    replica_sector_ids = tuple(
        max(sector_ids) for _, sector_ids in sorted(aliases_by_owner.items())
    )
    assert all(owners[sector_id] != sector_id for sector_id in replica_sector_ids)
    descriptors = tuple(
        ColorGroupDescriptor(
            group_id=group_id,
            helicity_key=(),
            sector_id=sector_id,
            word=color_plan.sectors[sector_id].color_words[0],
            helicity_weight=1.0,
        )
        for group_id, sector_id in enumerate(replica_sector_ids)
    )

    contraction = build_symmetric_group_color_contraction_plan(
        color_plan,
        descriptors,
        sector_owner_ids=owners,
    )

    block = contraction.symmetric_group_block
    assert block is not None
    assert block.channel_count == 6
    assert block.local_group_count == 12
    assert all(owners[sector_id] == sector_id for sector_id in block.local_sector_ids)
    assert sorted(contraction.destination_by_group) == list(range(12))
    assert (
        tuple(
            owners[replica_sector_ids[destination_id]]
            for destination_id in contraction.destination_by_group
        )
        == block.local_sector_ids
    )
    _assert_dense_reconstruction(color_plan, block)


def test_partition_retains_one_bounded_lazy_permutation_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    color_plan = _plan("d d~ > u u~ g g g g")
    owners, owner_sector_ids, _ = _owner_descriptors(color_plan)
    symmetric_group_module = importlib.import_module("pyamplicol.color.symmetric_group")
    original_unrank = symmetric_group_module._lexicographic_permutation_unrank
    materialized_ranks: list[int] = []

    def counted_unrank(degree: int, rank: int):
        materialized_ranks.append(rank)
        return original_unrank(degree, rank)

    monkeypatch.setattr(
        symmetric_group_module,
        "_lexicographic_permutation_unrank",
        counted_unrank,
    )
    partition = certify_symmetric_group_orbits(
        color_plan,
        owner_sector_ids,
        sector_owner_ids=owners,
    )

    assert materialized_ranks == []
    shared = partition.orbits[0].permutations
    assert all(orbit.permutations is shared for orbit in partition.orbits)
    assert not isinstance(shared, tuple)
    assert len(shared) == math.factorial(4)
    assert shared[0] == (0, 1, 2, 3)
    assert shared[-1] == (3, 2, 1, 0)
    assert materialized_ranks == [0, math.factorial(4) - 1]


@pytest.mark.parametrize("degree", range(7))
def test_lexicographic_rank_unrank_matches_itertools_order(degree: int) -> None:
    symmetric_group_module = importlib.import_module("pyamplicol.color.symmetric_group")
    expected = tuple(permutations(range(degree)))

    for rank, permutation in enumerate(expected):
        assert (
            symmetric_group_module._lexicographic_permutation_unrank(degree, rank)
            == permutation
        )
        assert (
            symmetric_group_module._lexicographic_permutation_rank(
                permutation,
                degree=degree,
            )
            == rank
        )


def test_lexicographic_rank_unrank_rejects_invalid_inputs() -> None:
    symmetric_group_module = importlib.import_module("pyamplicol.color.symmetric_group")

    with pytest.raises(ValueError, match="degree"):
        symmetric_group_module._lexicographic_permutation_unrank(11, 0)
    with pytest.raises(ValueError, match="out of bounds"):
        symmetric_group_module._lexicographic_permutation_unrank(4, math.factorial(4))
    with pytest.raises(ValueError, match="bijection"):
        symmetric_group_module._lexicographic_permutation_rank(
            (0, 1, 1, 3),
            degree=4,
        )
    with pytest.raises(ValueError, match="inconsistent"):
        symmetric_group_module._lexicographic_permutation_rank(
            (0, 1, 2),
            degree=4,
        )


@pytest.mark.parametrize("degree", (8, 10))
def test_high_degree_structure_uses_only_sampled_rank_unrank(degree: int) -> None:
    symmetric_group_module = importlib.import_module("pyamplicol.color.symmetric_group")
    group_order = math.factorial(degree)
    view = symmetric_group_module._LexicographicPermutations(degree)
    sampled_ranks = (0, 1, group_order // 2, group_order - 2, group_order - 1)

    assert len(view) == group_order
    for rank in sampled_ranks:
        permutation = view[rank]
        assert len(permutation) == degree
        assert sorted(permutation) == list(range(degree))
        assert (
            symmetric_group_module._lexicographic_permutation_rank(
                permutation,
                degree=degree,
            )
            == rank
        )

    relative_indices = symmetric_group_module._hermiticity_relative_indices(
        degree,
        channel_count=2,
    )
    equivariance_indices = symmetric_group_module._equivariance_sample_indices(degree)
    assert relative_indices == equivariance_indices
    assert len(relative_indices) == degree + 1
    selected = tuple(view[index] for index in relative_indices)
    assert selected[0] == tuple(range(degree))
    assert any(
        permutation != symmetric_group_module._inverse_permutation(permutation)
        for permutation in selected
    )


def test_large_degree_hermiticity_samples_are_deterministic_and_nontrivial() -> None:
    symmetric_group_module = importlib.import_module("pyamplicol.color.symmetric_group")
    relative_indices = symmetric_group_module._hermiticity_relative_indices(
        5,
        channel_count=2,
    )
    selected = tuple(
        symmetric_group_module._lexicographic_permutation_unrank(5, index)
        for index in relative_indices
    )

    assert selected[0] == tuple(range(5))
    for left_position in range(4):
        adjacent = list(range(5))
        adjacent[left_position], adjacent[left_position + 1] = (
            adjacent[left_position + 1],
            adjacent[left_position],
        )
        assert tuple(adjacent) in selected
    assert any(
        permutation != symmetric_group_module._inverse_permutation(permutation)
        for permutation in selected
    )
    assert len(relative_indices) < math.factorial(5)


@pytest.mark.parametrize("accuracy", ("nlc", "full"))
def test_single_trace_kernel_quotient_matches_every_direct_exact_row(
    monkeypatch: pytest.MonkeyPatch,
    accuracy: str,
) -> None:
    symmetric_group_module = importlib.import_module("pyamplicol.color.symmetric_group")
    color_plan = _plan("g g > g g g", accuracy=accuracy)
    owners, owner_sector_ids, descriptors = _owner_descriptors(color_plan)
    partition = certify_symmetric_group_orbits(
        color_plan,
        owner_sector_ids,
        sector_owner_ids=owners,
    )
    (orbit,) = partition.orbits
    identity_sector_id = orbit.sector_ids[0]
    direct = symmetric_group_module.exact_color_contraction_factor
    identity_left_calls = 0

    def counted(color_plan, left, right, **kwargs):
        nonlocal identity_left_calls
        if left.id == identity_sector_id:
            identity_left_calls += 1
        return direct(color_plan, left, right, **kwargs)

    monkeypatch.setattr(
        symmetric_group_module,
        "exact_color_contraction_factor",
        counted,
    )
    contraction = build_symmetric_group_color_contraction_plan(
        color_plan,
        descriptors,
        sector_owner_ids=owners,
    )
    block = contraction.symmetric_group_block
    assert block is not None
    sectors = {sector.id: sector for sector in color_plan.sectors}
    expected = tuple(
        direct(
            color_plan,
            sectors[identity_sector_id],
            sectors[sector_id],
            accuracy=accuracy,
            full_col_acc=20,
        )
        for sector_id in orbit.sector_ids
    )
    representatives = {
        symmetric_group_module._single_trace_diagonal_kernel_representative(item)
        for item in permutations(range(partition.degree))
    }
    assert block.kernel_exact_weights == expected
    assert (
        identity_left_calls == len(representatives) < math.factorial(partition.degree)
    )
    if accuracy == "nlc":
        assert any(value < 0 for value in expected)


def test_single_trace_kernel_quotient_stays_off_for_open_line_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    symmetric_group_module = importlib.import_module("pyamplicol.color.symmetric_group")
    color_plan = _plan("d d~ > u u~ g g")
    owners, _, descriptors = _owner_descriptors(color_plan)

    def unexpected(_permutation):
        raise AssertionError("single-trace quotient reached an open-line channel")

    monkeypatch.setattr(
        symmetric_group_module,
        "_single_trace_diagonal_kernel_representative",
        unexpected,
    )
    contraction = build_symmetric_group_color_contraction_plan(
        color_plan,
        descriptors,
        sector_owner_ids=owners,
    )
    block = contraction.symmetric_group_block
    assert block is not None
    assert block.channel_count == 6
    _assert_dense_reconstruction(color_plan, block)
