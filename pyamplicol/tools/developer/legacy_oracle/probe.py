# SPDX-License-Identifier: 0BSD
"""Probe output parsing, momentum ordering, and resolved-row helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any, cast

from .model import (
    FORTRAN_ATOL,
    FORTRAN_EVIDENCE_ATOL_DECIMAL,
    FORTRAN_EVIDENCE_RTOL_DECIMAL,
    FORTRAN_RTOL,
    LcRowPartition,
    LegacyOracleError,
    ProbeResult,
    SelectedFlowProbeResult,
)
from .processes import _permutation

_VALUE_RE = re.compile(
    r"^AMPICOL_COLOR_PROBE_VALUE\s+\S+\s+\d+\s+\d+\s+([+\-0-9.Ee]+)$",
    re.MULTILINE,
)
_COMPONENT_RE = re.compile(
    r"^AMPICOL_COLOR_PROBE_COMPONENTS\s+"
    r"([+\-0-9.Ee]+)\s+([+\-0-9.Ee]+)\s+([+\-0-9.Ee]+)$",
    re.MULTILINE,
)
# The pinned probe calls these "flows", but each value is an algebraic row
# partition of the LC contraction. Only the one-row case identifies a physical
# LC cell without additional information from the generator.
_LC_ROW_VALUE_RE = re.compile(
    r"^AMPICOL_COLOR_PROBE_FLOW_VALUE\s+(\d+)\s+([+\-0-9.Ee]+)$",
    re.MULTILINE,
)
_LC_ROW_PERMUTATION_RE = re.compile(
    r"^AMPICOL_COLOR_PROBE_FLOW_PERM\s+(\d+)\s+([0-9 ]+?)\s*$",
    re.MULTILINE,
)
_LC_ROW_SUM_RE = re.compile(
    r"^AMPICOL_COLOR_PROBE_FLOW_SUM\s+([+\-0-9.Ee]+)$",
    re.MULTILINE,
)
_INTEGER_LABELS = {
    "currents": "AMPICOL_COLOR_PROBE_CURRENTS",
    "vertices": "AMPICOL_COLOR_PROBE_VERTICES",
    "amplitudes": "AMPICOL_COLOR_PROBE_AMPLITUDES",
    "color_orders": "AMPICOL_COLOR_PROBE_COLOR_ORDERS",
}
_SELECTED_FLOW_VALUE_RE = re.compile(
    r"^AMPICOL_SELECTED_FLOW_PROBE_VALUE\s+(\d+)\s+(\d+)\s+"
    r"([+\-0-9.Ee]+)$",
    re.MULTILINE,
)
_SELECTED_FLOW_NORMALIZATION_RE = re.compile(
    r"^AMPICOL_SELECTED_FLOW_PROBE_NORMALIZATION\s+([+\-0-9.Ee]+)$",
    re.MULTILINE,
)
_SELECTED_FLOW_FIXED_VALUE_RE = re.compile(
    r"^AMPICOL_SELECTED_FLOW_PROBE_FIXED_HELICITY_VALUE\s+([+\-0-9.Ee]+)$",
    re.MULTILINE,
)
_SELECTED_FLOW_INTEGER_LABELS = {
    "amplitudes": "AMPICOL_SELECTED_FLOW_PROBE_AMPLITUDES",
    "color_factor": "AMPICOL_SELECTED_FLOW_PROBE_COLOR_FACTOR",
    "identical_factor": "AMPICOL_SELECTED_FLOW_PROBE_IDENTICAL_FACTOR",
    "singlet_vertices": "AMPICOL_SELECTED_FLOW_PROBE_SINGLET_VERTICES",
}


def _parse_probe_output(output: str) -> ProbeResult:
    value_matches = _VALUE_RE.findall(output)
    component_matches = _COMPONENT_RE.findall(output)
    if len(value_matches) != 1 or len(component_matches) != 1:
        raise LegacyOracleError(
            "Fortran color probe must emit exactly one matrix-element value and "
            "component record"
        )
    integers: dict[str, int] = {}
    for name, label in _INTEGER_LABELS.items():
        matches = re.findall(rf"^{label}\s+(\d+)$", output, re.MULTILINE)
        if len(matches) != 1:
            raise LegacyOracleError(
                f"Fortran color probe must emit exactly one {label} record"
            )
        integers[name] = int(matches[0])
    row_value_matches = _LC_ROW_VALUE_RE.findall(output)
    row_permutation_matches = _LC_ROW_PERMUTATION_RE.findall(output)
    row_sum_matches = _LC_ROW_SUM_RE.findall(output)
    row_value_text = {int(row): value for row, value in row_value_matches}
    row_permutations = {
        int(row): tuple(int(value) for value in permutation.split())
        for row, permutation in row_permutation_matches
    }
    try:
        value_float, value_decimal = _probe_number(
            value_matches[0], "matrix-element value"
        )
        component_pairs = tuple(
            _probe_number(value, "matrix-element component")
            for value in component_matches[0]
        )
    except (InvalidOperation, ValueError, OverflowError) as error:
        raise LegacyOracleError(
            "Fortran color probe emitted a malformed matrix-element number"
        ) from error
    if row_value_text or row_permutations or row_sum_matches:
        if (
            len(row_value_text) != len(row_value_matches)
            or len(row_permutations) != len(row_permutation_matches)
            or len(row_sum_matches) != 1
            or set(row_value_text) != set(row_permutations)
            or sorted(row_value_text) != list(range(1, len(row_value_text) + 1))
        ):
            raise LegacyOracleError(
                "Fortran color probe emitted incomplete LC row partitions"
            )
        try:
            row_numbers = {
                row: _probe_number(value, f"LC row {row} value")
                for row, value in row_value_text.items()
            }
            sum_number = _probe_number(row_sum_matches[0], "LC partition sum")
        except (InvalidOperation, ValueError, OverflowError) as error:
            raise LegacyOracleError(
                "Fortran color probe emitted a malformed LC partition number"
            ) from error
        lc_row_partitions = tuple(
            LcRowPartition(
                row,
                row_numbers[row][0],
                row_permutations[row],
                row_numbers[row][1],
            )
            for row in sorted(row_value_text)
        )
        lc_partition_sum, lc_partition_sum_decimal = sum_number
    else:
        lc_row_partitions = ()
        lc_partition_sum = None
        lc_partition_sum_decimal = None
    return ProbeResult(
        value=value_float,
        components=cast(
            tuple[float, float, float],
            tuple(value[0] for value in component_pairs),
        ),
        lc_row_partitions=lc_row_partitions,
        lc_partition_sum=lc_partition_sum,
        value_decimal=value_decimal,
        component_decimals=tuple(value[1] for value in component_pairs),
        lc_partition_sum_decimal=lc_partition_sum_decimal,
        **integers,
    )


def _parse_selected_flow_probe_output(output: str) -> SelectedFlowProbeResult:
    """Parse the indexed scalar emitted by the generated-library driver."""

    value_matches = _SELECTED_FLOW_VALUE_RE.findall(output)
    normalization_matches = _SELECTED_FLOW_NORMALIZATION_RE.findall(output)
    fixed_value_matches = _SELECTED_FLOW_FIXED_VALUE_RE.findall(output)
    if len(value_matches) != 1 or len(normalization_matches) != 1:
        raise LegacyOracleError(
            "Fortran selected-flow probe must emit exactly one value and "
            "normalization record"
        )
    if len(fixed_value_matches) > 1:
        raise LegacyOracleError(
            "Fortran selected-flow probe emitted duplicate fixed-helicity values"
        )
    group = int(value_matches[0][0])
    integral = int(value_matches[0][1])
    if group < 1 or integral < 1:
        raise LegacyOracleError(
            "Fortran selected-flow probe emitted a non-positive row identity"
        )

    process_pdgs = _parse_counted_integer_vector(
        output,
        "AMPICOL_SELECTED_FLOW_PROBE_PDGS",
    )
    color_order = _parse_counted_integer_vector(
        output,
        "AMPICOL_SELECTED_FLOW_PROBE_COLOR_ORDER",
    )
    has_helicity_record = "AMPICOL_SELECTED_FLOW_PROBE_HELICITIES" in output
    if bool(fixed_value_matches) != has_helicity_record:
        raise LegacyOracleError(
            "Fortran selected-flow probe emitted incomplete fixed-helicity evidence"
        )
    helicities = (
        _parse_counted_integer_vector(
            output,
            "AMPICOL_SELECTED_FLOW_PROBE_HELICITIES",
        )
        if has_helicity_record
        else ()
    )
    if len(process_pdgs) != len(color_order):
        raise LegacyOracleError(
            "Fortran selected-flow probe emitted inconsistent external metadata"
        )
    if helicities and len(helicities) != len(process_pdgs):
        raise LegacyOracleError(
            "Fortran selected-flow probe emitted a helicity vector with the "
            "wrong external width"
        )
    expected_order = tuple(range(1, len(process_pdgs) + 1))
    if tuple(sorted(color_order)) != expected_order:
        raise LegacyOracleError(
            "Fortran selected-flow probe color order is not a permutation of "
            f"1..{len(process_pdgs)}"
        )

    integers: dict[str, int] = {}
    for name, label in _SELECTED_FLOW_INTEGER_LABELS.items():
        matches = re.findall(rf"^{re.escape(label)}\s+(\d+)$", output, re.MULTILINE)
        if len(matches) != 1:
            raise LegacyOracleError(
                f"Fortran selected-flow probe must emit exactly one {label} record"
            )
        integers[name] = int(matches[0])
    if integers["amplitudes"] < 1:
        raise LegacyOracleError(
            "Fortran selected-flow probe emitted no generated amplitudes"
        )
    if integers["color_factor"] < 1 or integers["identical_factor"] < 1:
        raise LegacyOracleError(
            "Fortran selected-flow probe emitted a non-positive normalization factor"
        )

    try:
        value, value_decimal = _probe_number(
            value_matches[0][2],
            "selected-flow matrix-element value",
        )
        normalization, normalization_decimal = _probe_number(
            normalization_matches[0],
            "selected-flow normalization",
        )
        if fixed_value_matches:
            fixed_helicity_value, fixed_helicity_value_decimal = _probe_number(
                fixed_value_matches[0],
                "selected-flow fixed-helicity matrix-element value",
            )
        else:
            fixed_helicity_value = None
            fixed_helicity_value_decimal = None
    except (InvalidOperation, ValueError, OverflowError) as error:
        raise LegacyOracleError(
            "Fortran selected-flow probe emitted a malformed number"
        ) from error
    if normalization <= 0.0:
        raise LegacyOracleError(
            "Fortran selected-flow probe emitted a non-positive normalization"
        )

    return SelectedFlowProbeResult(
        value=value,
        group=group,
        integral=integral,
        process_pdgs=process_pdgs,
        color_order=color_order,
        normalization=normalization,
        helicities=helicities,
        fixed_helicity_value=fixed_helicity_value,
        value_decimal=value_decimal,
        normalization_decimal=normalization_decimal,
        fixed_helicity_value_decimal=fixed_helicity_value_decimal,
        **integers,
    )


def _parse_counted_integer_vector(output: str, label: str) -> tuple[int, ...]:
    matches = re.findall(
        rf"^{re.escape(label)}\s+(\d+)\s+([+\-0-9 ]+?)\s*$",
        output,
        re.MULTILINE,
    )
    if len(matches) != 1:
        raise LegacyOracleError(
            f"Fortran selected-flow probe must emit exactly one {label} record"
        )
    declared_count = int(matches[0][0])
    values = tuple(int(value) for value in matches[0][1].split())
    if declared_count < 1 or len(values) != declared_count:
        raise LegacyOracleError(
            f"Fortran selected-flow probe emitted an invalid {label} width"
        )
    return values


def _probe_number(value: str, context: str) -> tuple[float, Decimal]:
    decimal_value = Decimal(value)
    float_value = float(value)
    if not decimal_value.is_finite() or not math.isfinite(float_value):
        raise ValueError(f"{context} must be finite")
    return float_value, decimal_value


def _ordered_binary64_momenta(
    source_pdgs: Sequence[int],
    target_pdgs: Sequence[int],
    momenta: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float, float], ...]:
    if len(momenta) != len(source_pdgs):
        raise LegacyOracleError(
            f"received {len(momenta)} momentum rows for {len(source_pdgs)} external "
            "particles"
        )
    permutation = _permutation(source_pdgs, target_pdgs)
    ordered: list[tuple[float, float, float, float]] = []
    for source_index in permutation:
        vector = tuple(float(component) for component in momenta[source_index])
        if len(vector) != 4:
            raise LegacyOracleError("external momentum must contain four components")
        if not all(math.isfinite(component) for component in vector):
            raise LegacyOracleError("external momentum contains a non-finite binary64")
        ordered.append((vector[0], vector[1], vector[2], vector[3]))
    return tuple(ordered)


def _binary64_input_sha256(
    source_pdgs: Sequence[int],
    target_pdgs: Sequence[int],
    momenta: Sequence[Sequence[float]],
) -> str:
    """Hash post-permutation IEEE-754 values in canonical big-endian order."""

    ordered = _ordered_binary64_momenta(source_pdgs, target_pdgs, momenta)
    values = tuple(component for vector in ordered for component in vector)
    return hashlib.sha256(struct.pack(f">{len(values)}d", *values)).hexdigest()


def _resolved_case_value(cases: Mapping[str, Any], case: Mapping[str, Any]) -> Any:
    return (
        cases[str(case["resolved_from"])]["resolved"]
        if "resolved_from" in case
        else case["resolved"]
    )


def _momenta_case_value(cases: Mapping[str, Any], case: Mapping[str, Any]) -> Any:
    return (
        cases[str(case["momenta_from"])]["momenta"]
        if "momenta_from" in case
        else case["momenta"]
    )


def _helicities(identifier: str) -> tuple[int, ...]:
    if not identifier.startswith("h:"):
        raise LegacyOracleError(f"invalid physical helicity ID: {identifier!r}")
    return tuple(int(value) for value in identifier[2:].split(","))


def _close(actual: float, expected: float) -> bool:
    return math.isclose(
        actual,
        expected,
        rel_tol=FORTRAN_RTOL,
        abs_tol=FORTRAN_ATOL,
    )


def _decimal_value(value: float | Decimal) -> Decimal:
    decimal = value if isinstance(value, Decimal) else Decimal(str(value))
    if not decimal.is_finite():
        raise LegacyOracleError(f"Fortran oracle emitted non-finite value {value!r}")
    return decimal


def _canonical_decimal(value: float | Decimal) -> str:
    decimal = _decimal_value(value)
    if decimal == 0:
        return "0"
    rendered = format(decimal, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _probe_value_decimal(probe: ProbeResult) -> Decimal:
    return (
        probe.value_decimal
        if probe.value_decimal is not None
        else _decimal_value(probe.value)
    )


def _lc_row_value_decimal(partition: LcRowPartition) -> Decimal:
    return (
        partition.decimal_value
        if partition.decimal_value is not None
        else _decimal_value(partition.value)
    )


def _decimal_close(actual: Decimal, expected: Decimal) -> bool:
    scale = max(abs(actual), abs(expected))
    allowed = max(
        FORTRAN_EVIDENCE_ATOL_DECIMAL,
        FORTRAN_EVIDENCE_RTOL_DECIMAL * scale,
    )
    return abs(actual - expected) <= allowed


def _canonical_structural_zero(value: Decimal, *, context: str) -> Decimal:
    """Validate binary64 residue, then preserve an exact zero in evidence."""

    if not _decimal_close(value, Decimal(0)):
        raise LegacyOracleError(
            f"{context}: Fortran did not preserve structural zero ({value})"
        )
    return Decimal(0)


def _canonical_physics_output_sha256(
    summed: ProbeResult,
    helicity_results: Sequence[ProbeResult],
) -> str:
    """Hash deterministic physics output, excluding raw timer/logging text."""

    def stable_probe(probe: ProbeResult) -> dict[str, Any]:
        components = (
            probe.component_decimals
            if probe.component_decimals
            else tuple(_decimal_value(value) for value in probe.components)
        )
        return {
            "amplitudes": probe.amplitudes,
            "color_orders": probe.color_orders,
            "components": [_canonical_decimal(component) for component in components],
            "currents": probe.currents,
            "lc_partition_sum": (
                None
                if probe.lc_partition_sum is None
                else _canonical_decimal(
                    probe.lc_partition_sum_decimal
                    if probe.lc_partition_sum_decimal is not None
                    else probe.lc_partition_sum
                )
            ),
            "lc_row_partitions": [
                {
                    "permutation": list(partition.permutation),
                    "row": partition.row,
                    "value": _canonical_decimal(_lc_row_value_decimal(partition)),
                }
                for partition in probe.lc_row_partitions
            ],
            "value": _canonical_decimal(_probe_value_decimal(probe)),
            "vertices": probe.vertices,
        }

    payload = {
        "helicity_probes": [stable_probe(probe) for probe in helicity_results],
        "summed_probe": stable_probe(summed),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _resolved_single_lc_row(
    probe: ProbeResult,
    *,
    colors: Sequence[Mapping[str, Any]],
    source_to_row_permutation: Sequence[int],
    context: str,
) -> list[str]:
    if len(colors) != 1:
        raise LegacyOracleError(
            f"{context}: LC row partitions cannot resolve multiple physical flows"
        )
    if probe.lc_partition_sum is None or not probe.lc_row_partitions:
        raise LegacyOracleError(
            f"{context}: Fortran LC probe did not emit its row partition"
        )
    if probe.color_orders != 1 or len(probe.lc_row_partitions) != 1:
        raise LegacyOracleError(
            f"{context}: single-flow LC resolution requires exactly one Fortran "
            "color-order row"
        )
    partition_sum = (
        probe.lc_partition_sum_decimal
        if probe.lc_partition_sum_decimal is not None
        else _decimal_value(probe.lc_partition_sum)
    )
    aggregate = _probe_value_decimal(probe)
    row_sum = sum(
        (_lc_row_value_decimal(partition) for partition in probe.lc_row_partitions),
        Decimal(0),
    )
    if not _decimal_close(partition_sum, row_sum):
        raise LegacyOracleError(
            f"{context}: Fortran LC row value is {row_sum}, but the reported row "
            f"sum is {partition_sum}"
        )
    if not _decimal_close(partition_sum, aggregate):
        raise LegacyOracleError(
            f"{context}: Fortran LC row sum {partition_sum} does not match "
            f"aggregate {aggregate}"
        )

    colors_by_word = {
        tuple(int(position) for position in color["word"]): color for color in colors
    }
    if len(colors_by_word) != len(colors):
        raise LegacyOracleError(f"{context}: fixture contains duplicate LC words")
    values_by_id: dict[str, str] = {}
    for partition in probe.lc_row_partitions:
        try:
            source_word = tuple(
                source_to_row_permutation[position - 1] + 1
                for position in partition.permutation
            )
        except IndexError as error:
            raise LegacyOracleError(
                f"{context}: Fortran row {partition.row} contains an invalid position"
            ) from error
        if any(position < 1 for position in partition.permutation):
            raise LegacyOracleError(
                f"{context}: Fortran row {partition.row} contains an invalid position"
            )
        color = colors_by_word.get(source_word)
        if color is None:
            raise LegacyOracleError(
                f"{context}: Fortran row {partition.row} maps to unknown fixture word "
                f"{source_word}"
            )
        color_id = str(color["id"])
        if color_id in values_by_id:
            raise LegacyOracleError(
                f"{context}: several Fortran rows map to color {color_id}"
            )
        values_by_id[color_id] = _canonical_decimal(_lc_row_value_decimal(partition))
    expected_ids = {str(color["id"]) for color in colors}
    if set(values_by_id) != expected_ids:
        missing = sorted(expected_ids - values_by_id.keys())
        raise LegacyOracleError(
            f"{context}: Fortran probe did not cover LC colors {missing}"
        )
    return [values_by_id[str(color["id"])] for color in colors]
