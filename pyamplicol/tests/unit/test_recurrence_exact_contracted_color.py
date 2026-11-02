# SPDX-License-Identifier: 0BSD
"""Exact contracted-color recurrence tests over compact binary rows."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import astuple, replace
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import permutations
from pathlib import Path

import pytest

from pyamplicol._internal.versions import (
    RECURRENCE_HELICITY_SELECTOR_COMPANION_RUNTIME_CAPABILITY,
    SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY,
)
from pyamplicol.api.errors import ArtifactError, CompatibilityError, EvaluationError
from pyamplicol.artifacts.manifest import ArtifactManifest, PayloadRecord
from pyamplicol.color import (
    ColorContractionEntry,
    ColorContractionPlan,
    ColorContractionTemplateEntry,
    SymmetricGroupColorContractionBlock,
)
from pyamplicol.generation.recurrence_color import (
    RecurrenceColorCodecError,
    encode_recurrence_color_contraction,
)
from pyamplicol.generation.recurrence_columnar import ExactComplexRationalV1
from pyamplicol.runtime.recurrence_exact import _executor as executor_module
from pyamplicol.runtime.recurrence_exact._color import (
    RECURRENCE_COLOR_CONTRACTION_CODEC_ABI,
    RECURRENCE_CONTRACTED_COLOR_CAPABILITY,
    _contract_color_amplitudes,
    _decode_recurrence_color_contraction,
    _load_recurrence_color_contraction,
)
from pyamplicol.runtime.recurrence_exact._executor import RecurrenceExactExecutor
from pyamplicol.runtime.recurrence_exact._plan import (
    _RecurrenceExactPlan,
    _validate_execution,
)
from pyamplicol.runtime.recurrence_exact._plan_v2 import (
    RECURRENCE_DIRECT_RUNTIME_CAPABILITY,
    RECURRENCE_EXACT_SECTIONS_ABI,
    RECURRENCE_RUNTIME_LAYOUT_V2_ABI,
    _AmplitudeDestination,
    _parse_exact_sections,
    _RecurrenceExactSectionsV1,
    _ReplayTarget,
    _ResolvedHelicity,
    _SourceStateAssignment,
)

_HEADER = struct.Struct("<8s14I7Q")
_ENTRY = struct.Struct("<IIdddI")
_EXACT_FACTOR_BYTES = 64
_U32 = struct.Struct("<I")


def _contracted_execution_with_capabilities(
    capabilities: tuple[str, ...],
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "kind": "pyamplicol-runtime-recurrence-execution",
        "key": "synthetic_contracted",
        "recurrence_plan_abi": "pyamplicol-recurrence-plan-v2",
        "runtime_layout_abi": "pyamplicol-recurrence-runtime-layout-v2",
        "recurrence_summary": {"lc_flow_layout": "contracted-color-union"},
        "required_runtime_capabilities": capabilities,
        "color_accuracy": "full",
    }


def test_exact_execution_permits_helicity_companion_and_exact_fft_fallback() -> None:
    core = (
        RECURRENCE_DIRECT_RUNTIME_CAPABILITY,
        RECURRENCE_CONTRACTED_COLOR_CAPABILITY,
    )
    _validate_execution(
        _contracted_execution_with_capabilities(
            (*core, RECURRENCE_HELICITY_SELECTOR_COMPANION_RUNTIME_CAPABILITY)
        ),
        "synthetic_contracted",
    )
    _validate_execution(
        _contracted_execution_with_capabilities(
            (*core, SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY)
        ),
        "synthetic_contracted",
    )

    for capabilities in (
        (core[0], RECURRENCE_HELICITY_SELECTOR_COMPANION_RUNTIME_CAPABILITY),
        (*core, "rusticol.recurrence-unknown.v1"),
    ):
        with pytest.raises(
            CompatibilityError,
            match="unsupported recurrence runtime capability contract",
        ):
            _validate_execution(
                _contracted_execution_with_capabilities(capabilities),
                "synthetic_contracted",
            )


def _color_payload(
    *,
    storage: str,
    sector_count: int,
    component_count: int,
    entries: tuple[tuple[int, int, float, float, float], ...],
    destinations: tuple[int, ...] | None = None,
    ordered_groups: tuple[int, ...] | None = None,
    group_sector_ids: tuple[int, ...] | None = None,
    group_component_ids: tuple[int, ...] | None = None,
    sector_owner_ids: tuple[int, ...] | None = None,
    color_accuracy: str = "full",
    includes_color_factor: bool = True,
) -> bytes:
    if group_sector_ids is None and group_component_ids is None:
        group_count = sector_count * component_count
        sectors = tuple(group_id // component_count for group_id in range(group_count))
        components = tuple(
            group_id % component_count for group_id in range(group_count)
        )
    elif group_sector_ids is not None and group_component_ids is not None:
        group_count = len(group_sector_ids)
        sectors = group_sector_ids
        components = group_component_ids
    else:
        raise AssertionError("test payload needs both group coordinate maps")
    repeated = storage == "repeated"
    ordered = ordered_groups or tuple(range(group_count))
    destination_map = destinations or tuple(range(group_count))
    destination_count = max(destination_map, default=-1) + 1
    exact_factors = tuple(
        (
            Fraction(str(weight_re)) * Fraction(str(symmetry_factor)),
            Fraction(str(weight_im)) * Fraction(str(symmetry_factor)),
        )
        for _, _, weight_re, weight_im, symmetry_factor in entries
    )
    active_sectors = set(sectors)
    owners = (
        sector_owner_ids
        if sector_owner_ids is not None
        else tuple(
            sector_id if sector_id in active_sectors else 0xFFFF_FFFF
            for sector_id in range(sector_count)
        )
    )
    payload_size = (
        len(entries) * _ENTRY.size
        + len(exact_factors) * _EXACT_FACTOR_BYTES
        + len(ordered) * _U32.size
        + len(destination_map) * _U32.size
        + len(sectors) * _U32.size
        + len(components) * _U32.size
        + len(owners) * _U32.size
    )
    header = _HEADER.pack(
        b"PACRCLR3",
        3,
        _HEADER.size,
        2 if repeated else 1,
        2 if color_accuracy == "full" else 1,
        1 if includes_color_factor else 0,
        group_count,
        sector_count,
        component_count,
        group_count // component_count if repeated else 0,
        destination_count,
        0,
        0,
        _ENTRY.size,
        _EXACT_FACTOR_BYTES,
        len(entries),
        len(exact_factors),
        0,
        0,
        len(entries) * component_count if repeated else len(entries),
        len(owners),
        payload_size,
    )
    encoded_factors = tuple(
        value.to_bytes(16, "little", signed=True)
        for real, imag in exact_factors
        for value in (
            real.numerator,
            real.denominator,
            imag.numerator,
            imag.denominator,
        )
    )
    return b"".join(
        (
            header,
            *(_ENTRY.pack(*entry, index) for index, entry in enumerate(entries)),
            *encoded_factors,
            *(_U32.pack(value) for value in ordered),
            *(_U32.pack(value) for value in destination_map),
            *(_U32.pack(value) for value in sectors),
            *(_U32.pack(value) for value in components),
            *(_U32.pack(value) for value in owners),
        )
    )


def test_compact_repeated_rows_expand_without_decimal_strings() -> None:
    payload = _color_payload(
        storage="repeated",
        sector_count=2,
        component_count=2,
        entries=((0, 1, 0.1, 0.0, 2.0),),
        ordered_groups=(2, 0, 3, 1),
        destinations=(0, 1, 2, 3),
        group_sector_ids=(0, 1, 0, 1),
        group_component_ids=(1, 1, 0, 0),
    )
    contraction = _decode_recurrence_color_contraction(payload)

    rows = tuple(contraction.runtime_entries())
    assert tuple(
        (row.left_destination_id, row.right_destination_id) for row in rows
    ) == ((2, 3), (0, 1))
    assert rows[0].coefficient_re == Decimal("0.2")
    assert rows[0].coefficient_im == Decimal(0)
    assert tuple(row.component_id for row in rows) == (0, 1)


def test_compact_repeated_rows_support_a_rectangular_active_sector_subset() -> None:
    payload = _color_payload(
        storage="repeated",
        sector_count=4,
        component_count=2,
        entries=((0, 1, 0.25, 0.0, 2.0),),
        ordered_groups=(0, 2, 1, 3),
        group_sector_ids=(1, 3, 1, 3),
        group_component_ids=(0, 0, 1, 1),
    )
    contraction = _decode_recurrence_color_contraction(payload)

    assert contraction.sector_count == 4
    assert tuple(
        (row.left_destination_id, row.right_destination_id)
        for row in contraction.runtime_entries()
    ) == ((0, 1), (2, 3))


def test_compact_expanded_color_rows_reject_cross_helicity_contraction() -> None:
    payload = _color_payload(
        storage="expanded",
        sector_count=2,
        component_count=2,
        entries=((0, 1, 1.0, 0.0, 1.0),),
        group_sector_ids=(0, 1),
        group_component_ids=(0, 1),
    )
    with pytest.raises(ArtifactError, match="mixes helicity components"):
        _decode_recurrence_color_contraction(payload)


def test_compact_expanded_color_rows_support_sparse_sector_component_cells() -> None:
    payload = _color_payload(
        storage="expanded",
        sector_count=3,
        component_count=2,
        entries=((0, 1, 0.5, 0.0, 2.0),),
        destinations=(2, 0, 1),
        group_sector_ids=(0, 2, 1),
        group_component_ids=(1, 1, 0),
    )
    contraction = _decode_recurrence_color_contraction(payload)

    assert contraction.group_count == 3
    assert contraction.sector_count == 3
    assert contraction.component_count == 2
    assert contraction.group_sector_ids == (0, 2, 1)
    assert contraction.group_component_ids == (1, 1, 0)
    row = next(contraction.runtime_entries())
    assert row.component_id == 1
    assert (row.left_destination_id, row.right_destination_id) == (2, 0)


@pytest.mark.parametrize(
    ("sectors", "components", "message"),
    (
        ((0, 2), (0, 0), "out of bounds"),
        ((0, 1), (0, 2), "out of bounds"),
        ((0, 0), (1, 1), "not unique"),
    ),
)
def test_compact_color_rows_reject_malformed_group_coordinate_maps(
    sectors: tuple[int, ...],
    components: tuple[int, ...],
    message: str,
) -> None:
    payload = _color_payload(
        storage="expanded",
        sector_count=2,
        component_count=2,
        entries=(),
        group_sector_ids=sectors,
        group_component_ids=components,
    )
    with pytest.raises(ArtifactError, match=message):
        _decode_recurrence_color_contraction(payload)


def test_compact_color_rows_authenticate_sector_aliases_and_structural_zeros() -> None:
    valid = _color_payload(
        storage="expanded",
        sector_count=4,
        component_count=1,
        entries=((0, 1, 0.5, 0.0, 2.0),),
        group_sector_ids=(0, 2),
        group_component_ids=(0, 0),
        sector_owner_ids=(0, 0, 2, 0xFFFF_FFFF),
    )
    contraction = _decode_recurrence_color_contraction(valid)
    assert contraction.owner_by_sector == (0, 0, 2, 0xFFFF_FFFF)

    invalid = bytearray(valid)
    owner_offset = _HEADER.size + _ENTRY.size + _EXACT_FACTOR_BYTES + 4 * 2 * _U32.size
    _U32.pack_into(invalid, owner_offset + _U32.size, 2)
    with pytest.raises(ArtifactError, match="invalid owner"):
        _decode_recurrence_color_contraction(bytes(invalid))


def test_compact_color_rows_reject_exact_and_binary64_coefficient_drift() -> None:
    payload = bytearray(
        _color_payload(
            storage="expanded",
            sector_count=1,
            component_count=1,
            entries=((0, 0, 0.5, 0.0, 2.0),),
        )
    )
    exact_factor_offset = _HEADER.size + _ENTRY.size
    payload[exact_factor_offset : exact_factor_offset + 16] = (2).to_bytes(
        16,
        "little",
        signed=True,
    )
    with pytest.raises(ArtifactError, match="disagrees with its exact factor"):
        _decode_recurrence_color_contraction(bytes(payload))


def test_compact_repeated_rows_reject_mixed_local_color_rows() -> None:
    payload = _color_payload(
        storage="repeated",
        sector_count=2,
        component_count=2,
        entries=((0, 1, 1.0, 0.0, 1.0),),
        ordered_groups=(0, 2, 1, 3),
    )
    with pytest.raises(ArtifactError, match="every component exactly once"):
        _decode_recurrence_color_contraction(payload)


def test_compact_color_decoder_rejects_v1_without_compatibility() -> None:
    payload = bytearray(
        _color_payload(
            storage="expanded",
            sector_count=1,
            component_count=1,
            entries=((0, 0, 1.0, 0.0, 1.0),),
        )
    )
    payload[:8] = b"PACRCLR1"
    struct.pack_into("<I", payload, 8, 1)
    with pytest.raises(ArtifactError, match="unsupported"):
        _decode_recurrence_color_contraction(bytes(payload))


def test_color_encoder_accepts_sparse_expanded_coordinates() -> None:
    plan = ColorContractionPlan(
        color_accuracy="full",
        supported=True,
        reason=None,
        group_count=3,
        entries=(ColorContractionEntry(0, 1, 0.5, symmetry_factor=2.0),),
    )
    payload = encode_recurrence_color_contraction(
        plan,
        sector_count=3,
        component_count=2,
        ordered_group_ids=(2, 0, 1),
        destination_by_group=(2, 0, 1),
        group_sector_ids=(0, 2, 1),
        group_component_ids=(1, 1, 0),
        sector_owner_ids=(0, 1, 2),
        exact_coefficients=(ExactComplexRationalV1(1),),
        destination_count=3,
    )
    contraction = _decode_recurrence_color_contraction(payload)
    assert contraction.group_count == 3
    assert contraction.group_sector_ids == (0, 2, 1)


def test_color_encoder_emits_canonical_symmetric_group_kernel_rows() -> None:
    block = SymmetricGroupColorContractionBlock(
        degree=2,
        component_count=1,
        component_group_ids=(0, 1, 2),
        local_sector_ids=(0, 1, 2),
        channel_cosets=((0, 1),),
        kernel_entries=(
            ColorContractionTemplateEntry(0, 0, 3.0),
            ColorContractionTemplateEntry(0, 1, 1.0),
        ),
        kernel_exact_weights=(Fraction(3), Fraction(1)),
        residual_entries=(
            ColorContractionTemplateEntry(0, 2, 0.25, symmetry_factor=2.0),
            ColorContractionTemplateEntry(1, 2, 0.0, symmetry_factor=2.0),
            ColorContractionTemplateEntry(2, 2, 5.0),
        ),
        residual_exact_weights=(Fraction(1, 4), Fraction(0), Fraction(5)),
        residual_local_group_indices=(2,),
    )
    plan = ColorContractionPlan(
        color_accuracy="full",
        supported=True,
        reason=None,
        group_count=3,
        entries=(),
        symmetric_group_block=block,
        destination_by_group=(2, 0, 1),
    )
    exact_values = (
        Fraction(3),
        Fraction(1),
        Fraction(1, 2),
        Fraction(0),
        Fraction(5),
    )
    exact = tuple(
        ExactComplexRationalV1(value.numerator, value.denominator)
        for value in exact_values
    )
    payload = encode_recurrence_color_contraction(
        plan,
        sector_count=3,
        component_count=1,
        ordered_group_ids=(0, 1, 2),
        destination_by_group=(2, 0, 1),
        group_sector_ids=(0, 1, 2),
        group_component_ids=(0, 0, 0),
        sector_owner_ids=(0, 1, 2),
        exact_coefficients=exact,
        destination_count=3,
    )

    header = _HEADER.unpack_from(payload)
    assert header[0] == b"PACRCLR3"
    assert header[1:4] == (3, _HEADER.size, 3)
    assert header[11:13] == (3, 2)
    assert header[15:20] == (5, 5, 1, 2, 5)
    rows = tuple(
        _ENTRY.unpack_from(payload, _HEADER.size + index * _ENTRY.size)[:5]
        for index in range(5)
    )
    assert rows == (
        (0, 0, 3.0, 0.0, 1.0),
        (0, 1, 1.0, 0.0, 1.0),
        (0, 2, 0.25, 0.0, 2.0),
        (1, 2, 0.0, 0.0, 2.0),
        (2, 2, 5.0, 0.0, 1.0),
    )
    assert struct.unpack_from("<2I", payload, len(payload) - 2 * _U32.size) == (
        0,
        1,
    )

    contraction = _decode_recurrence_color_contraction(payload)
    assert contraction.storage == "convolution-kernels"
    assert contraction.factorization_kind == "symmetric-group-fourier"
    assert len(tuple(contraction.runtime_entries())) == 7
    assert _contract_color_amplitudes(
        contraction,
        (
            (Decimal(1), Decimal(2)),
            (Decimal(3), Decimal(-1)),
            (Decimal(-2), Decimal(1)),
        ),
        (0, 0, 0),
    ) == {0: Decimal("76.5")}

    with pytest.raises(RecurrenceColorCodecError, match="certified convolution plan"):
        encode_recurrence_color_contraction(
            plan,
            sector_count=3,
            component_count=1,
            ordered_group_ids=(0, 1, 2),
            destination_by_group=(2, 0, 1),
            group_sector_ids=(0, 1, 2),
            group_component_ids=(0, 0, 0),
            sector_owner_ids=(0, 1, 2),
            exact_coefficients=(*exact[:-1], ExactComplexRationalV1(6)),
            destination_count=3,
        )

    object.__setattr__(
        block,
        "residual_entries",
        (
            block.residual_entries[1],
            block.residual_entries[0],
            block.residual_entries[2],
        ),
    )
    object.__setattr__(
        block,
        "residual_exact_weights",
        (
            block.residual_exact_weights[1],
            block.residual_exact_weights[0],
            block.residual_exact_weights[2],
        ),
    )

    with pytest.raises(RecurrenceColorCodecError, match="do not exhaust"):
        encode_recurrence_color_contraction(
            plan,
            sector_count=3,
            component_count=1,
            ordered_group_ids=(0, 1, 2),
            destination_by_group=(2, 0, 1),
            group_sector_ids=(0, 1, 2),
            group_component_ids=(0, 0, 0),
            sector_owner_ids=(0, 1, 2),
            exact_coefficients=(*exact[:2], exact[3], exact[2], exact[4]),
            destination_count=3,
        )


def test_exact_fft_fallback_preserves_cross_channels_residuals_and_helicity_maps() -> (
    None
):
    group = tuple(permutations(range(3)))
    order = len(group)
    local_count, component_count = 13, 2
    ordered = tuple(reversed(range(local_count * component_count)))
    destinations = tuple((index * 7) % len(ordered) for index in range(len(ordered)))
    sectors, components = [0] * len(ordered), [0] * len(ordered)
    for local in range(local_count):
        for component in range(component_count):
            group_id = ordered[local * component_count + component]
            sectors[group_id], components[group_id] = local, component
    # The diagonal kernels obey inversion symmetry, but the cross kernel
    # deliberately does not: reversing L^{-1}R would change the answer.
    kernels = (
        tuple(Fraction(value, 7) for value in (10, 2, 3, 5, 5, 4)),
        tuple(Fraction(value, 11) for value in (1, 2, 3, 4, 6, 7)),
        tuple(Fraction(value, 13) for value in (8, 1, 2, 3, 3, 4)),
    )
    channel_pairs = ((0, 0), (0, 1), (1, 1))
    kernel_entries = tuple(
        ColorContractionTemplateEntry(
            left * order,
            right * order + relative,
            float(value),
            symmetry_factor=1.0 if left == right else 2.0,
        )
        for (left, right), kernel in zip(channel_pairs, kernels, strict=True)
        for relative, value in enumerate(kernel)
    )
    residual = tuple(Fraction(index + 2, 17) for index in range(local_count))
    residual_entries = tuple(
        ColorContractionTemplateEntry(
            index,
            12,
            float(value),
            symmetry_factor=1.0 if index == 12 else 2.0,
        )
        for index, value in enumerate(residual)
    )
    weights = tuple(value for kernel in kernels for value in kernel) + residual
    entries = kernel_entries + residual_entries
    exact = tuple(
        ExactComplexRationalV1(value.numerator, value.denominator)
        for entry, weight in zip(entries, weights, strict=True)
        for value in (weight * int(entry.symmetry_factor),)
    )
    block = SymmetricGroupColorContractionBlock(
        degree=3,
        component_count=component_count,
        component_group_ids=ordered,
        local_sector_ids=tuple(range(local_count)),
        channel_cosets=(tuple(range(6)), tuple(range(6, 12))),
        kernel_entries=kernel_entries,
        kernel_exact_weights=weights[:18],
        residual_entries=residual_entries,
        residual_exact_weights=residual,
        residual_local_group_indices=(12,),
        hermiticity_check_mode="full",
        hermiticity_relative_indices=tuple(range(6)),
    )
    payload = encode_recurrence_color_contraction(
        ColorContractionPlan(
            color_accuracy="full",
            supported=True,
            reason=None,
            group_count=len(ordered),
            entries=(),
            symmetric_group_block=block,
            destination_by_group=destinations,
        ),
        sector_count=local_count,
        component_count=component_count,
        ordered_group_ids=ordered,
        destination_by_group=destinations,
        group_sector_ids=tuple(sectors),
        group_component_ids=tuple(components),
        sector_owner_ids=tuple(range(local_count)),
        exact_coefficients=exact,
        destination_count=len(ordered),
    )
    contraction = _decode_recurrence_color_contraction(payload)
    amplitudes = [(Decimal(0), Decimal(0))] * len(ordered)
    helicities = [0] * len(ordered)
    expected = {}
    for component in range(component_count):
        local_values = tuple(
            ((-1) ** local * (local + 1), (2 * local + 1) * (component + 1))
            for local in range(local_count)
        )
        for local, value in enumerate(local_values):
            destination = destinations[ordered[local * component_count + component]]
            amplitudes[destination] = tuple(Decimal(part) for part in value)
            helicities[destination] = 10 + component

        def real_product(left, right, values=local_values):
            a, b = values[left], values[right]
            return a[0] * b[0] + a[1] * b[1]

        result = Fraction(0)
        for (left_channel, right_channel), kernel in zip(
            channel_pairs, kernels, strict=True
        ):
            for left_index, left in enumerate(group):
                for right_index, right in enumerate(group):
                    relative = group.index(tuple(left.index(value) for value in right))
                    result += (
                        kernel[relative]
                        * (1 if left_channel == right_channel else 2)
                        * real_product(
                            left_channel * 6 + left_index,
                            right_channel * 6 + right_index,
                        )
                    )
        result += sum(
            weight * (1 if local == 12 else 2) * real_product(local, 12)
            for local, weight in enumerate(residual)
        )
        expected[10 + component] = result
    with localcontext() as context:
        context.prec = 70
        actual = _contract_color_amplitudes(contraction, amplitudes, helicities)
        for helicity, value in expected.items():
            reference = Decimal(value.numerator) / Decimal(value.denominator)
            assert abs(actual[helicity] - reference) < Decimal("1e-60")
        assert _contract_color_amplitudes(
            contraction, amplitudes, helicities, {11}
        ) == {11: actual[11]}


def test_color_encoder_rejects_duplicate_group_coordinates() -> None:
    plan = ColorContractionPlan(
        color_accuracy="nlc",
        supported=True,
        reason=None,
        group_count=2,
        entries=(),
    )
    with pytest.raises(RecurrenceColorCodecError, match="duplicate"):
        encode_recurrence_color_contraction(
            plan,
            sector_count=2,
            component_count=2,
            ordered_group_ids=(0, 1),
            destination_by_group=(0, 1),
            group_sector_ids=(0, 0),
            group_component_ids=(1, 1),
            sector_owner_ids=(0, 1),
            exact_coefficients=(),
            destination_count=2,
        )


def test_compact_color_payload_is_authenticated_against_manifest_and_summary(
    tmp_path: Path,
) -> None:
    payload = _color_payload(
        storage="repeated",
        sector_count=2,
        component_count=2,
        entries=(
            (0, 0, 1.0, 0.0, 1.0),
            (0, 1, 0.5, 0.0, 2.0),
            (1, 1, 1.0, 0.0, 1.0),
        ),
    )
    digest = hashlib.sha256(payload).hexdigest()
    relative = "processes/process/recurrence-color.bin"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    manifest = ArtifactManifest(
        root=tmp_path,
        kind="pyamplicol-process",
        artifact_id="artifact",
        created_utc="2026-01-01T00:00:00Z",
        producer={},
        model={},
        configuration={},
        processes=(),
        default_process_id=None,
        runtime={},
        payloads=(
            PayloadRecord(
                path=relative,
                role="evaluator-state",
                media_type="application/octet-stream",
                size_bytes=len(payload),
                sha256=digest,
                executable=False,
                process_id="process",
            ),
        ),
        dependencies=(),
        extensions={},
    )
    reference = {
        "abi": RECURRENCE_COLOR_CONTRACTION_CODEC_ABI,
        "path": "recurrence-color.bin",
        "size_bytes": len(payload),
        "sha256": digest,
        "color_accuracy": "full",
        "storage": "repeated",
        "includes_color_factor": True,
        "group_count": 4,
        "sector_count": 2,
        "active_sector_count": 2,
        "component_count": 2,
        "destination_count": 4,
        "entry_count": 3,
        "logical_entry_count": 6,
        "semantic_digest": digest,
    }
    loaded = _load_recurrence_color_contraction(
        artifact_root=tmp_path,
        process_id="process",
        execution_path="processes/process/execution.json",
        execution={"runtime_metadata": {"color_contraction": reference}},
        manifest=manifest,
    )
    assert loaded is not None
    assert loaded.logical_entry_count == 6

    corrupted = {**reference, "logical_entry_count": 7}
    with pytest.raises(ArtifactError, match="bounded summary"):
        _load_recurrence_color_contraction(
            artifact_root=tmp_path,
            process_id="process",
            execution_path="processes/process/execution.json",
            execution={"runtime_metadata": {"color_contraction": corrupted}},
            manifest=manifest,
        )


def test_exact_color_contraction_uses_complex_hermitian_product() -> None:
    contraction = _decode_recurrence_color_contraction(
        _color_payload(
            storage="expanded",
            sector_count=2,
            component_count=1,
            entries=((0, 1, 2.0, 0.5, 2.0),),
        )
    )
    amplitudes = (
        (Decimal(1), Decimal(2)),
        (Decimal(3), Decimal(4)),
    )
    result = _contract_color_amplitudes(contraction, amplitudes, (0, 0))
    # A0 conj(A1) = 11 + 2i and c = 4 + i after symmetry folding.
    assert result == {0: Decimal(42)}
    assert _contract_color_amplitudes(contraction, amplitudes, (0, 0), {1}) == {}


def _contracted_plan() -> _RecurrenceExactPlan:
    resolved = (
        _ResolvedHelicity(0, 0, 0, 0, 2, 0, 2, 0),
        _ResolvedHelicity(2, 0, 2, 1, 2, 0, 2, 0),
    )
    destinations = tuple(
        _AmplitudeDestination(
            destination_id,
            destination_id,
            destination_id // 2,
            destination_id % 2,
            1,
            0,
        )
        for destination_id in range(4)
    )
    sections = _RecurrenceExactSectionsV1(
        process_id="contracted",
        strategy="contracted-color-union",
        semantic_digest="0" * 64,
        runtime_layout_digest="1" * 64,
        current_arena_components=1,
        amplitude_destination_count=4,
        parameter_value_count=0,
        external_source_count=2,
        currents=(),
        sources=(),
        contributions=(),
        finalizations=(),
        closures=(),
        row_groups=(),
        momentum_forms=(),
        momentum_terms=(),
        replay_targets=(),
        source_permutations=(),
        replay_momentum_signs=(),
        replay_helicity_map=(),
        amplitude_destinations=destinations,
        resolved_helicities=resolved,
        source_state_assignments=(
            _SourceStateAssignment(0, 0),
            _SourceStateAssignment(1, 0),
            _SourceStateAssignment(0, 1),
            _SourceStateAssignment(1, 1),
        ),
        source_dispatch_variants=(),
        source_embeddings=(),
        source_projections=(),
        resolved_source_selections=(),
        public_helicities=(0, 0, 1, 1),
        exact_factors=(),
        public_flow_ids=(),
        executors=(),
    )
    contraction = _decode_recurrence_color_contraction(
        _color_payload(
            storage="repeated",
            sector_count=2,
            component_count=2,
            entries=(
                (0, 0, 1.0, 0.0, 1.0),
                (0, 1, 0.5, 0.0, 2.0),
                (1, 1, 1.0, 0.0, 1.0),
            ),
        )
    )
    return _RecurrenceExactPlan(
        sections=sections,
        kernels={},
        executors={},
        executor_exact_kernel_ids={},
        executor_parent_permutations={},
        source_templates={},
        initial_source_slots=frozenset(),
        executor_couplings={},
        prepared_defaults=(),
        parameter_projection=(),
        parameter_derivation=None,
        color_contraction=contraction,
    )


def _contracted_replay_plan() -> _RecurrenceExactPlan:
    plan = _contracted_plan()
    sections = replace(
        plan.sections,
        amplitude_destination_count=2,
        replay_targets=(
            _ReplayTarget(0, 0, 0, 2, 0, 2, 0, 1, 0),
            _ReplayTarget(1, 0, 2, 2, 2, 2, 0, 1, 0),
        ),
        source_permutations=(0, 1, 1, 0),
        replay_momentum_signs=(1, 1, 1, 1),
        replay_helicity_map=(0, 1, 1, 0),
        amplitude_destinations=plan.sections.amplitude_destinations[:2],
    )
    return replace(plan, sections=sections)


def _native_sections(sections: _RecurrenceExactSectionsV1) -> dict[str, object]:
    return {
        "abi": RECURRENCE_EXACT_SECTIONS_ABI,
        "runtime_layout_abi": RECURRENCE_RUNTIME_LAYOUT_V2_ABI,
        "process_id": sections.process_id,
        "strategy": sections.strategy,
        "semantic_digest": sections.semantic_digest,
        "runtime_layout_digest": sections.runtime_layout_digest,
        "counts": (
            sections.current_arena_components,
            sections.amplitude_destination_count,
            sections.parameter_value_count,
            sections.external_source_count,
        ),
        "currents": [],
        "sources": [],
        "contributions": [],
        "finalizations": [],
        "closures": [],
        "row_groups": [],
        "momentum_forms": [],
        "momentum_terms": [],
        "replay_targets": [astuple(row) for row in sections.replay_targets],
        "source_permutations": list(sections.source_permutations),
        "replay_momentum_signs": list(sections.replay_momentum_signs),
        "replay_helicity_map": list(sections.replay_helicity_map),
        "amplitude_destinations": [
            astuple(row) for row in sections.amplitude_destinations
        ],
        "resolved_helicities": [astuple(row) for row in sections.resolved_helicities],
        "source_state_assignments": [
            astuple(row) for row in sections.source_state_assignments
        ],
        "source_dispatch_variants": [],
        "source_embeddings": [],
        "source_projections": [],
        "resolved_source_selections": [],
        "public_helicities": list(sections.public_helicities),
        "exact_factors": [],
        "public_flow_ids": list(sections.public_flow_ids),
        "executors": [],
    }


def test_contracted_native_section_adapter_accepts_fixed_source_grid() -> None:
    sections = _contracted_plan().sections
    raw = _native_sections(sections)
    parsed = _parse_exact_sections(raw, sections.process_id)
    assert parsed.strategy == "contracted-color-union"
    assert parsed.amplitude_destinations == sections.amplitude_destinations

    raw["public_flow_ids"] = [0]
    with pytest.raises(ArtifactError, match="public color-flow axis"):
        _parse_exact_sections(raw, sections.process_id)


def test_contracted_native_section_adapter_accepts_internal_replay() -> None:
    sections = _contracted_replay_plan().sections
    parsed = _parse_exact_sections(_native_sections(sections), sections.process_id)

    assert parsed.public_flow_ids == ()
    assert parsed.replay_targets == sections.replay_targets
    assert parsed.replay_helicity_map == (0, 1, 1, 0)

    malformed = _native_sections(sections)
    malformed["source_permutations"] = [0, 0, 1, 0]
    with pytest.raises(ArtifactError, match="complete source bijection"):
        _parse_exact_sections(malformed, sections.process_id)

    malformed = _native_sections(sections)
    malformed["replay_helicity_map"] = [0, 0, 1, 0]
    with pytest.raises(ArtifactError, match="complete bijection"):
        _parse_exact_sections(malformed, sections.process_id)


def test_contracted_exact_replay_reconstructs_dense_physical_destinations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = object.__new__(RecurrenceExactExecutor)
    executor._plan = _contracted_replay_plan()
    executor._permutation = (0, 1)
    executor._physics = {
        "helicities": [
            {"id": "h:0,0", "values": [0, 0]},
            {"id": "h:1,1", "values": [1, 1]},
        ]
    }
    replayed = {
        0: ((Decimal(1), Decimal(0)), (Decimal(2), Decimal(0))),
        1: ((Decimal(3), Decimal(0)), (Decimal(4), Decimal(0))),
    }
    monkeypatch.setattr(
        executor_module,
        "_evaluate_replay_point",
        lambda _plan, _point, target, *_args: replayed[target.public_flow_id],
    )

    amplitudes = executor._contracted_amplitudes(object(), (), 70)
    destination_helicities = executor._contracted_destination_helicity_map()

    assert amplitudes == (
        (Decimal(1), Decimal(0)),
        (Decimal(2), Decimal(0)),
        (Decimal(4), Decimal(0)),
        (Decimal(3), Decimal(0)),
    )
    assert destination_helicities == (0, 1, 0, 1)
    assert _contract_color_amplitudes(
        executor._plan.color_contraction,
        amplitudes,
        destination_helicities,
    ) == {0: Decimal(21), 1: Decimal(19)}


def test_contracted_exact_resolved_output_and_selector_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = object.__new__(RecurrenceExactExecutor)
    executor._plan = _contracted_plan()
    executor._physics = {
        "extensions": {"normalization": {"average_factor": 2}},
        "external_particles": [{}, {}],
        "helicities": [
            {
                "id": "h:0,0",
                "values": [0, 0],
                "computed": True,
                "structural_zero": False,
                "coefficient": 1,
            },
            {
                "id": "h:1,1",
                "values": [1, 1],
                "computed": True,
                "structural_zero": False,
                "coefficient": 2,
            },
        ],
        "color_components": [
            {
                "kind": "contracted-color",
                "id": "color:contracted",
                "index": 0,
                "description": "contracted",
            }
        ],
        "color_accuracy": "full",
    }
    executor._permutation = (0, 1)
    executor._native_runtime = object()
    (
        executor._helicity_representative,
        executor._helicity_orbit_members,
    ) = executor._helicity_reduction_indices()
    executor._replay_by_color = ()
    executor._destination_helicities = ()
    executor._union_destination_by_color = ()
    executor._union_helicity_by_physics = ()
    executor._contracted_destination_helicity = (
        executor._contracted_destination_helicity_map()
    )
    monkeypatch.setattr(
        executor_module,
        "_prepare_points",
        lambda *_: (((Decimal(1),) * 4, (Decimal(2),) * 4),),
    )
    monkeypatch.setattr(
        executor_module,
        "_runtime_state",
        lambda _: {"model_parameter_values": [], "normalization_factor": "0.5"},
    )
    monkeypatch.setattr(
        executor_module,
        "_evaluate_contracted_point",
        lambda *_: (
            (Decimal(1), Decimal(0)),
            (Decimal(2), Decimal(0)),
            (Decimal(3), Decimal(0)),
            (Decimal(4), Decimal(0)),
        ),
    )

    resolved = executor.evaluate_resolved(
        (((1, 0, 0, 1), (1, 0, 0, -1)),),
        helicities=None,
        color_flows=None,
        precision=70,
    )
    assert resolved.values == (
        (
            (Decimal("6.5"),),
            (Decimal(28),),
        ),
    )
    assert resolved.total() == (Decimal("34.5"),)
    assert resolved.color_ids == ("color:contracted",)

    selected = executor.evaluate_resolved(
        (((1, 0, 0, 1), (1, 0, 0, -1)),),
        helicities=("h:1,1",),
        color_flows=None,
        precision=70,
    )
    assert selected.values == (((Decimal(28),),),)

    with pytest.raises(EvaluationError, match="does not expose a color-flow selector"):
        executor.evaluate_resolved(
            (((1, 0, 0, 1), (1, 0, 0, -1)),),
            helicities=None,
            color_flows=("color:contracted",),
            precision=70,
        )


def test_exact_helicity_reduction_indexes_physical_alias_orbits() -> None:
    executor = object.__new__(RecurrenceExactExecutor)
    executor._physics = {
        "helicities": [
            {
                "id": "h:-1,+1",
                "computed": True,
                "structural_zero": False,
                "representative_id": "h:-1,+1",
            },
            {
                "id": "h:+1,-1",
                "computed": False,
                "structural_zero": False,
                "representative_id": "h:-1,+1",
            },
            {
                "id": "h:+1,+1",
                "computed": False,
                "structural_zero": True,
                "representative_id": "h:+1,+1",
            },
        ]
    }

    representatives, members = executor._helicity_reduction_indices()

    assert representatives == (0, 0, 2)
    assert members == ((0, 1), (), ())
