# SPDX-License-Identifier: 0BSD
"""Exact trace and NLC algebra used by color contraction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction

from .contraction_types import NC


def _eval_trace(
    traces: tuple[tuple[int, ...], ...],
    *,
    coeff: Fraction = Fraction(1),
) -> float:
    terms = _simplify_trace_terms(((coeff, traces),))
    return _eval_nc_terms(terms)


def _eval_trace_exact(
    traces: tuple[tuple[int, ...], ...],
    *,
    coeff: Fraction = Fraction(1),
) -> Fraction:
    """Evaluate a trace product at ``Nc`` without passing through binary64."""

    terms = _simplify_trace_terms(((coeff, traces),))
    return _eval_nc_terms_exact(terms)


def _simplify_trace_terms(
    initial: Iterable[tuple[Fraction, tuple[tuple[int, ...], ...]]],
) -> Mapping[int, Fraction]:
    terms: list[tuple[Fraction, tuple[tuple[int, ...], ...]]] = [
        (coeff, tuple(tuple(trace) for trace in traces))
        for coeff, traces in initial
        if coeff
    ]
    guard = 0
    while True:
        guard += 1
        if guard > 100000:
            raise RuntimeError("colour trace simplification did not converge")
        terms = _simplify_tr1_tr0(terms)
        if not terms:
            return {}
        if all(not traces for _coeff, traces in terms):
            result: dict[int, Fraction] = {}
            for coeff, _traces in terms:
                result[0] = result.get(0, Fraction(0)) + coeff
            return result
        next_terms, changed = _trace_pair_simplify(terms)
        terms = _simplify_tr1_tr0(next_terms)
        next_terms, changed_dup = _trace_duplicate_simplify(terms)
        terms = next_terms
        if not changed and not changed_dup and any(traces for _coeff, traces in terms):
            raise RuntimeError(f"cannot simplify colour traces: {terms[:3]}")


def _simplify_trace_terms_nc_power(
    initial: Iterable[tuple[Fraction, int, tuple[tuple[int, ...], ...]]],
) -> Mapping[int, Fraction]:
    terms: list[tuple[Fraction, int, tuple[tuple[int, ...], ...]]] = [
        (coeff, power, tuple(tuple(trace) for trace in traces))
        for coeff, power, traces in initial
        if coeff
    ]
    guard = 0
    while True:
        guard += 1
        if guard > 100000:
            raise RuntimeError("colour trace simplification did not converge")
        terms = _simplify_tr1_tr0_nc_power(terms)
        if not terms:
            return {}
        if all(not traces for _coeff, _power, traces in terms):
            result: dict[int, Fraction] = {}
            for coeff, power, _traces in terms:
                result[power] = result.get(power, Fraction(0)) + coeff
            return {power: coeff for power, coeff in result.items() if coeff}
        next_terms, changed = _trace_pair_simplify_nc_power(terms)
        terms = _simplify_tr1_tr0_nc_power(next_terms)
        next_terms, changed_dup = _trace_duplicate_simplify_nc_power(terms)
        terms = _combine_nc_power_terms(next_terms)
        if (
            not changed
            and not changed_dup
            and any(traces for _coeff, _power, traces in terms)
        ):
            raise RuntimeError(f"cannot simplify colour traces: {terms[:3]}")


def _simplify_tr1_tr0_nc_power(
    terms: list[tuple[Fraction, int, tuple[tuple[int, ...], ...]]],
) -> list[tuple[Fraction, int, tuple[tuple[int, ...], ...]]]:
    result: list[tuple[Fraction, int, tuple[tuple[int, ...], ...]]] = []
    for coeff, power, traces in terms:
        if any(len(trace) == 1 for trace in traces):
            continue
        extra_power = sum(1 for trace in traces if len(trace) == 0)
        remaining = tuple(trace for trace in traces if len(trace) != 0)
        result.append((coeff, power + extra_power, remaining))
    return _combine_nc_power_terms(result)


def _trace_pair_simplify_nc_power(
    terms: list[tuple[Fraction, int, tuple[tuple[int, ...], ...]]],
) -> tuple[list[tuple[Fraction, int, tuple[tuple[int, ...], ...]]], bool]:
    result: list[tuple[Fraction, int, tuple[tuple[int, ...], ...]]] = []
    changed = False
    for coeff, power, traces in terms:
        replaced = False
        for first_index, first in enumerate(traces):
            for second_index in range(first_index + 1, len(traces)):
                second = traces[second_index]
                found = _first_common_position(first, second)
                if found is None:
                    continue
                first_pos, second_pos = found
                combined = (
                    first[:first_pos]
                    + second[second_pos + 1 :]
                    + second[:second_pos]
                    + first[first_pos + 1 :]
                )
                rest = tuple(
                    trace
                    for index, trace in enumerate(traces)
                    if index not in {first_index, second_index}
                )
                result.append((coeff, power, (combined, *rest)))
                reduced_first = first[:first_pos] + first[first_pos + 1 :]
                reduced_second = second[:second_pos] + second[second_pos + 1 :]
                result.append(
                    (
                        -coeff,
                        power - 1,
                        tuple(
                            trace
                            if index not in {first_index, second_index}
                            else (
                                reduced_first
                                if index == first_index
                                else reduced_second
                            )
                            for index, trace in enumerate(traces)
                        ),
                    )
                )
                changed = True
                replaced = True
                break
            if replaced:
                break
        if not replaced:
            result.append((coeff, power, traces))
    return result, changed


def _trace_duplicate_simplify_nc_power(
    terms: list[tuple[Fraction, int, tuple[tuple[int, ...], ...]]],
) -> tuple[list[tuple[Fraction, int, tuple[tuple[int, ...], ...]]], bool]:
    result: list[tuple[Fraction, int, tuple[tuple[int, ...], ...]]] = []
    changed = False
    for coeff, power, traces in terms:
        replaced = False
        for trace_index, trace in enumerate(traces):
            positions = _first_duplicate_positions(trace)
            if positions is None:
                continue
            first_pos, second_pos = positions
            a = trace[:first_pos]
            b = trace[first_pos + 1 : second_pos]
            c = trace[second_pos + 1 :]
            rest = tuple(
                item for index, item in enumerate(traces) if index != trace_index
            )
            result.append((coeff, power, (a + c, b, *rest)))
            result.append((-coeff, power - 1, (a + b + c, *rest)))
            changed = True
            replaced = True
            break
        if not replaced:
            result.append((coeff, power, traces))
    return result, changed


def _combine_nc_power_terms(
    terms: Iterable[tuple[Fraction, int, tuple[tuple[int, ...], ...]]],
) -> list[tuple[Fraction, int, tuple[tuple[int, ...], ...]]]:
    combined: dict[tuple[int, tuple[tuple[int, ...], ...]], Fraction] = {}
    for coeff, power, traces in terms:
        if not coeff:
            continue
        key = (power, tuple(sorted(tuple(trace) for trace in traces)))
        combined[key] = combined.get(key, Fraction(0)) + coeff
    return [
        (coeff, power, traces) for (power, traces), coeff in combined.items() if coeff
    ]


def _simplify_tr1_tr0(
    terms: list[tuple[Fraction, tuple[tuple[int, ...], ...]]],
) -> list[tuple[Fraction, tuple[tuple[int, ...], ...]]]:
    result: list[tuple[Fraction, tuple[tuple[int, ...], ...]]] = []
    for coeff, traces in terms:
        if any(len(trace) == 1 for trace in traces):
            continue
        power = sum(1 for trace in traces if len(trace) == 0)
        remaining = tuple(trace for trace in traces if len(trace) != 0)
        result.append((coeff * (NC**power), remaining))
    return [(coeff, traces) for coeff, traces in result if coeff]


def _trace_pair_simplify(
    terms: list[tuple[Fraction, tuple[tuple[int, ...], ...]]],
) -> tuple[list[tuple[Fraction, tuple[tuple[int, ...], ...]]], bool]:
    result: list[tuple[Fraction, tuple[tuple[int, ...], ...]]] = []
    changed = False
    for coeff, traces in terms:
        replaced = False
        for first_index, first in enumerate(traces):
            for second_index in range(first_index + 1, len(traces)):
                second = traces[second_index]
                found = _first_common_position(first, second)
                if found is None:
                    continue
                first_pos, second_pos = found
                combined = (
                    first[:first_pos]
                    + second[second_pos + 1 :]
                    + second[:second_pos]
                    + first[first_pos + 1 :]
                )
                rest = tuple(
                    trace
                    for index, trace in enumerate(traces)
                    if index not in {first_index, second_index}
                )
                result.append((coeff, (combined, *rest)))
                reduced_first = first[:first_pos] + first[first_pos + 1 :]
                reduced_second = second[:second_pos] + second[second_pos + 1 :]
                result.append(
                    (
                        -coeff / NC,
                        tuple(
                            trace
                            if index not in {first_index, second_index}
                            else (
                                reduced_first
                                if index == first_index
                                else reduced_second
                            )
                            for index, trace in enumerate(traces)
                        ),
                    )
                )
                changed = True
                replaced = True
                break
            if replaced:
                break
        if not replaced:
            result.append((coeff, traces))
    return result, changed


def _trace_duplicate_simplify(
    terms: list[tuple[Fraction, tuple[tuple[int, ...], ...]]],
) -> tuple[list[tuple[Fraction, tuple[tuple[int, ...], ...]]], bool]:
    result: list[tuple[Fraction, tuple[tuple[int, ...], ...]]] = []
    changed = False
    for coeff, traces in terms:
        replaced = False
        for trace_index, trace in enumerate(traces):
            positions = _first_duplicate_positions(trace)
            if positions is None:
                continue
            first_pos, second_pos = positions
            a = trace[:first_pos]
            b = trace[first_pos + 1 : second_pos]
            c = trace[second_pos + 1 :]
            rest = tuple(
                item for index, item in enumerate(traces) if index != trace_index
            )
            result.append((coeff, (a + c, b, *rest)))
            result.append((-coeff / NC, (a + b + c, *rest)))
            changed = True
            replaced = True
            break
        if not replaced:
            result.append((coeff, traces))
    return result, changed


def _first_common_position(
    left: Sequence[int],
    right: Sequence[int],
) -> tuple[int, int] | None:
    for left_index, label in enumerate(left):
        for right_index, other in enumerate(right):
            if label == other:
                return left_index, right_index
    return None


def _first_duplicate_positions(trace: Sequence[int]) -> tuple[int, int] | None:
    seen: dict[int, int] = {}
    for index, label in enumerate(trace):
        if label in seen:
            return seen[label], index
        seen[label] = index
    return None


def _eval_nc_terms(
    terms: Mapping[int, Fraction],
    *,
    min_power: int | None = None,
) -> float:
    value = 0.0
    for power, coeff in terms.items():
        if min_power is not None and power < min_power:
            continue
        value += float(coeff) * (float(NC) ** power)
    return value


def _eval_nc_terms_exact(
    terms: Mapping[int, Fraction],
    *,
    min_power: int | None = None,
) -> Fraction:
    """Evaluate an exact Laurent polynomial in ``Nc`` as a ``Fraction``."""

    value = Fraction(0)
    nc = Fraction(NC)
    for power, coeff in terms.items():
        if min_power is not None and power < min_power:
            continue
        value += coeff * nc**power
    return value


def _check_nlc(jper: tuple[int, ...], iper: tuple[int, ...]) -> int:
    n = len(iper)
    if n == 0:
        return 99
    i1 = next((i for i in range(n) if jper[i] != iper[i]), n)
    if i1 == n:
        return 99
    try:
        i2 = next(i for i in range(i1 + 1, n) if jper[i] == iper[i1])
    except StopIteration:
        return 0
    offset = 0
    while (
        i2 + offset < n and i1 + offset < n and jper[i2 + offset] == iper[i1 + offset]
    ):
        offset += 1
    i3 = i2 + offset - 1
    i4 = i1 + offset - 1
    if i4 + 1 >= n:
        return 0
    try:
        i5 = next(i for i in range(i1, n) if jper[i] == iper[i4 + 1])
    except StopIteration:
        return 0
    if i5 > i3:
        return 0
    sign = 1
    if i1 > i5 - 1:
        left_len = i4 - i1
        right_len = i2 - 1 - i5
        if (left_len == 0 and right_len == 0) or (
            n > 0
            and (
                (left_len == 0 and right_len == n - 3)
                or (left_len == n - 3 and right_len == 0)
            )
        ):
            sign = -1
        elif left_len == 0 or right_len == 0:
            return 0
        elif left_len + right_len <= max(n - 4, 0):
            pass
        else:
            return 0
    itemp = jper[:i1] + jper[i2 : i3 + 1] + jper[i5:i2] + jper[i1:i5] + jper[i3 + 1 :]
    return sign if itemp == iper else 0


def _check_nlc_one_open_line(jper: tuple[int, ...], iper: tuple[int, ...]) -> int:
    """Port AmpliCol's one-open-line NLC topology check."""

    n = len(iper)
    if len(jper) != n:
        return 0
    i1 = next((i for i in range(n) if jper[i] != iper[i]), n)
    if i1 >= n:
        return 99
    try:
        i2 = next(i for i in range(i1 + 1, n) if jper[i] == iper[i1])
    except StopIteration:
        return 0
    offset = 0
    while (
        i2 + offset < n and i1 + offset < n and jper[i2 + offset] == iper[i1 + offset]
    ):
        offset += 1
    i3 = i2 + offset - 1
    i4 = i1 + offset - 1
    if i4 + 1 >= n:
        return 0
    try:
        i5 = next(i for i in range(i1, n) if jper[i] == iper[i4 + 1])
    except StopIteration:
        return 0
    if i5 > i3:
        return 0
    sign = 1
    if i1 > i5 - 1:
        left_len = i4 - i1
        right_len = i2 - 1 - i5
        if left_len == 0 and right_len == 0:
            sign = -1
        elif left_len == 0 or right_len == 0:
            return 0
        elif left_len + right_len <= n - 4:
            pass
        else:
            sign = 1
    itemp = jper[:i1] + jper[i2 : i3 + 1] + jper[i5:i2] + jper[i1:i5] + jper[i3 + 1 :]
    return sign if itemp == iper else 0
