# SPDX-License-Identifier: 0BSD
"""Adversarial contracts for ordered multi-open-line recurrence color state."""

from __future__ import annotations

import hashlib

import pytest

from pyamplicol import _rusticol
from pyamplicol.generation.recurrence_columnar import (
    ExactComplexRationalV1,
    RecurrenceBuilderInputV1,
    RecurrenceBuilderLogicalInputV1,
    RecurrenceExternalLegV1,
    RecurrenceLCOpenStringV1,
    RecurrenceNormalizationV1,
    RecurrencePhysicalLCSectorV1,
    RecurrencePublicLCFlowV1,
    RecurrenceSemanticDigestV1,
    RecurrenceSemanticTemplateReferenceV1,
    RecurrenceSourceStateV1,
    build_recurrence_builder_input_v1,
)


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _source_leg(source_slot: int) -> RecurrenceExternalLegV1:
    state_template_id = 100 + source_slot
    source_template_id = 200 + source_slot
    return RecurrenceExternalLegV1(
        source_slot=source_slot,
        public_label=source_slot + 1,
        physical_pdg=source_slot + 1,
        outgoing_pdg=source_slot + 1,
        is_initial=source_slot < 2,
        is_fermionic=True,
        source_states=(
            RecurrenceSourceStateV1(
                state_index=0,
                public_helicity=-1 if source_slot % 2 == 0 else 1,
                chirality=0,
                spin_state=0,
                current_state_template_id=state_template_id,
                source_template_id=source_template_id,
            ),
        ),
        momentum_mask=1 << source_slot,
        support_mask=1 << source_slot,
    )


def _three_open_line_logical_input() -> RecurrenceBuilderLogicalInputV1:
    """Two sectors differing only by the order of passive open-line blocks."""

    line_a = RecurrenceLCOpenStringV1(0, 1)
    line_b = RecurrenceLCOpenStringV1(2, 3)
    line_c = RecurrenceLCOpenStringV1(4, 5)
    sectors = (
        RecurrencePhysicalLCSectorV1(
            sector_id=0,
            public_id="flow:1,2|3,4|5,6",
            kind="open-lines",
            closure_source_slot=5,
            closure_proof_algorithm="canonical-lc-closure-anchor-v2",
            closure_proof_digest=_sha256("closure:abc"),
            open_strings=(line_a, line_b, line_c),
            word_source_slots=(0, 1, 2, 3, 4, 5),
            support_mask=0b11_1111,
        ),
        RecurrencePhysicalLCSectorV1(
            sector_id=1,
            public_id="flow:3,4|1,2|5,6",
            kind="open-lines",
            closure_source_slot=5,
            closure_proof_algorithm="canonical-lc-closure-anchor-v2",
            closure_proof_digest=_sha256("closure:bac"),
            open_strings=(line_b, line_a, line_c),
            word_source_slots=(2, 3, 0, 1, 4, 5),
            support_mask=0b11_1111,
        ),
    )
    references = tuple(
        RecurrenceSemanticTemplateReferenceV1(
            kind=kind,
            template_id=base + source_slot,
            semantic_digest=_sha256(f"{kind}:{source_slot}"),
            prepared_kernel_id=(300 + source_slot if kind == "source" else None),
        )
        for kind, base in (("current-state", 100), ("source", 200))
        for source_slot in range(6)
    )
    return RecurrenceBuilderLogicalInputV1(
        process_id="synthetic_three_open_lines",
        layout="all-flow-union",
        semantic_digests=tuple(
            RecurrenceSemanticDigestV1(role, _sha256(role))
            for role in (
                "process",
                "model-catalog",
                "prepared-catalog",
                "color-plan",
            )
        ),
        external_legs=tuple(_source_leg(slot) for slot in range(6)),
        physical_sectors=sectors,
        public_flows=tuple(
            RecurrencePublicLCFlowV1(
                flow_id=sector.sector_id,
                public_id=sector.public_id,
                construction_sector_id=sector.sector_id,
                word_source_slots=sector.word_source_slots,
                source_slot_permutation=tuple(range(6)),
            )
            for sector in sectors
        ),
        semantic_template_references=references,
        normalization=RecurrenceNormalizationV1(
            ExactComplexRationalV1(1),
            "synthetic-adversarial-v1",
            _sha256("normalization"),
        ),
        process_support_mask=0b11_1111,
    )


def _table_rows(
    value: RecurrenceBuilderInputV1,
    table_name: str,
    *columns: str,
) -> tuple[tuple[int, ...], ...]:
    table = next(table for table in value.tables if table.name == table_name)
    return tuple(
        tuple(int(table.column(column)[row]) for column in columns)
        for row in range(table.row_count)
    )


def test_columnar_input_preserves_passive_open_line_forest_order() -> None:
    """Python must not canonicalize independent open-line blocks by sorting."""

    encoded = build_recurrence_builder_input_v1(_three_open_line_logical_input())

    rows = _table_rows(
        encoded,
        "lc_open_strings",
        "sector_id",
        "ordinal",
        "fundamental_source_slot",
        "antifundamental_source_slot",
    )
    assert rows == (
        (0, 0, 0, 1),
        (0, 1, 2, 3),
        (0, 2, 4, 5),
        (1, 0, 2, 3),
        (1, 1, 0, 1),
        (1, 2, 4, 5),
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the native inspection API does not yet expose ordered dynamic LC forests; "
        "it must return each state's ordered component words and active index"
    ),
)
def test_native_schedule_keeps_permuted_passive_forests_distinct() -> None:
    """A and B passive blocks may alias only with an exact permutation witness.

    Required production contract: a bounded native schedule-inspection operation
    accepting authenticated process/template inputs and returning ordered dynamic
    LC component words, active indices, and any exact permutation certificate.
    Merely returning ``dynamic_color_state_count`` cannot prove this invariant.
    """

    inspect_schedule = vars(_rusticol)["_inspect_recurrence_schedule_v1"]
    inspection = inspect_schedule(
        build_recurrence_builder_input_v1(_three_open_line_logical_input())
    )
    states = inspection["dynamic_lc_color_states"]
    abc = ((0, 1), (2, 3), (4, 5))
    bac = ((2, 3), (0, 1), (4, 5))
    by_components = {tuple(map(tuple, row["components"])): row for row in states}
    assert abc in by_components
    assert bac in by_components
    assert by_components[abc]["state_id"] != by_components[bac]["state_id"]
    assert by_components[abc].get("permutation_certificate") is None
    assert by_components[bac].get("permutation_certificate") is None
