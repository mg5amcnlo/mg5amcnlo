# SPDX-License-Identifier: 0BSD
"""Representation-based color-factor algorithms."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from functools import lru_cache

from .contraction_trace import (
    _check_nlc,
    _check_nlc_one_open_line,
    _eval_nc_terms,
    _eval_nc_terms_exact,
    _eval_trace,
    _eval_trace_exact,
    _simplify_trace_terms,
    _simplify_trace_terms_nc_power,
)
from .contraction_types import NC, ColorGroupDescriptor
from .plan import GenericColorPlan, LCColorSector, LCOpenColorLine


def _pure_adjoint_color_factors(
    left: LCColorSector,
    right: LCColorSector,
    n_ord: int,
    full_col_acc: int,
) -> tuple[float, float, float]:
    return (
        _pure_adjoint_color_factor(
            left,
            right,
            n_ord,
            accuracy="lc",
            full_col_acc=full_col_acc,
        ),
        _pure_adjoint_color_factor(
            left,
            right,
            n_ord,
            accuracy="nlc",
            full_col_acc=full_col_acc,
        ),
        _pure_adjoint_color_factor(
            left,
            right,
            n_ord,
            accuracy="full",
            full_col_acc=full_col_acc,
        ),
    )


def _pure_adjoint_color_factors_exact(
    left: LCColorSector,
    right: LCColorSector,
    n_ord: int,
    full_col_acc: int,
) -> tuple[Fraction, Fraction, Fraction]:
    return (
        _pure_adjoint_color_factor_exact(
            left,
            right,
            n_ord,
            accuracy="lc",
            full_col_acc=full_col_acc,
        ),
        _pure_adjoint_color_factor_exact(
            left,
            right,
            n_ord,
            accuracy="nlc",
            full_col_acc=full_col_acc,
        ),
        _pure_adjoint_color_factor_exact(
            left,
            right,
            n_ord,
            accuracy="full",
            full_col_acc=full_col_acc,
        ),
    )


def _pure_adjoint_color_factor(
    left: LCColorSector,
    right: LCColorSector,
    n_ord: int,
    *,
    accuracy: str,
    full_col_acc: int,
) -> float:
    iper = _coloured_word(left)
    jper = _coloured_word(right)
    if n_ord == 0:
        return 1.0 if not iper and not jper else 0.0
    if accuracy == "lc":
        return float(NC**n_ord) if iper == jper else 0.0
    if accuracy == "nlc":
        if n_ord == 2:
            # Tr(Ta Tb)=delta_ab, whose norm Nc^2-1 is already exact at
            # NLC. The n>=3 shortcut below would incorrectly count two
            # independent U(1) subtractions for the same identity tensor.
            return float(NC**2 - 1)
        if iper == jper:
            return float(NC**n_ord - n_ord * NC ** (n_ord - 2))
        return float(_check_nlc(tuple(jper), tuple(iper)) * NC ** (n_ord - 2))
    if accuracy != "full":
        return 0.0
    relative = _relative_adjoint_permutation(iper, jper)
    if relative is not None:
        return _pure_adjoint_full_factor_by_relative_permutation(
            relative,
            n_ord,
            full_col_acc,
        )
    return _pure_adjoint_full_factor_uncached(
        tuple(iper),
        tuple(jper),
        n_ord,
        full_col_acc,
    )


def _pure_adjoint_color_factor_exact(
    left: LCColorSector,
    right: LCColorSector,
    n_ord: int,
    *,
    accuracy: str,
    full_col_acc: int,
) -> Fraction:
    iper = _coloured_word(left)
    jper = _coloured_word(right)
    if n_ord == 0:
        return Fraction(1 if not iper and not jper else 0)
    nc = Fraction(NC)
    if accuracy == "lc":
        return nc**n_ord if iper == jper else Fraction(0)
    if accuracy == "nlc":
        if n_ord == 2:
            return nc**2 - 1
        if iper == jper:
            return nc**n_ord - n_ord * nc ** (n_ord - 2)
        return Fraction(_check_nlc(tuple(jper), tuple(iper))) * nc ** (n_ord - 2)
    if accuracy != "full":
        return Fraction(0)
    relative = _relative_adjoint_permutation(iper, jper)
    if relative is not None:
        return _pure_adjoint_full_factor_exact_by_relative_permutation(
            relative,
            n_ord,
            full_col_acc,
        )
    return _pure_adjoint_full_factor_exact_uncached(
        tuple(iper),
        tuple(jper),
        n_ord,
        full_col_acc,
    )


def _relative_adjoint_permutation(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> tuple[int, ...] | None:
    """Canonicalize a trace pair under simultaneous adjoint relabelling."""

    if len(left) != len(right) or len(set(left)) != len(left):
        return None
    position = {label: index for index, label in enumerate(left)}
    try:
        relative = tuple(position[label] for label in right)
    except KeyError:
        return None
    if len(set(relative)) != len(relative):
        return None
    return relative


@lru_cache(maxsize=65536)
def _pure_adjoint_full_factor_by_relative_permutation(
    relative: tuple[int, ...],
    n_ord: int,
    full_col_acc: int,
) -> float:
    canonical = tuple(range(len(relative)))
    return _pure_adjoint_full_factor_uncached(
        canonical,
        relative,
        n_ord,
        full_col_acc,
    )


@lru_cache(maxsize=65536)
def _pure_adjoint_full_factor_exact_by_relative_permutation(
    relative: tuple[int, ...],
    n_ord: int,
    full_col_acc: int,
) -> Fraction:
    canonical = tuple(range(len(relative)))
    return _pure_adjoint_full_factor_exact_uncached(
        canonical,
        relative,
        n_ord,
        full_col_acc,
    )


def _pure_adjoint_full_factor_uncached(
    iper: tuple[int, ...],
    jper: tuple[int, ...],
    n_ord: int,
    full_col_acc: int,
) -> float:
    full_terms = _simplify_trace_terms(
        ((Fraction(1), (tuple(iper), tuple(reversed(jper)))),)
    )
    return _eval_nc_terms(
        full_terms,
        min_power=max(n_ord - 2 * full_col_acc, 0),
    )


def _pure_adjoint_full_factor_exact_uncached(
    iper: tuple[int, ...],
    jper: tuple[int, ...],
    n_ord: int,
    full_col_acc: int,
) -> Fraction:
    full_terms = _simplify_trace_terms(
        ((Fraction(1), (tuple(iper), tuple(reversed(jper)))),)
    )
    return _eval_nc_terms_exact(
        full_terms,
        min_power=max(n_ord - 2 * full_col_acc, 0),
    )


def _one_open_line_color_factors(
    left: LCColorSector,
    right: LCColorSector,
    n_ord: int,
) -> tuple[float, float, float]:
    iper = _coloured_word(left)
    jper = _coloured_word(right)
    lc = float(NC ** (n_ord - 1)) if iper == jper else 0.0
    full = _eval_trace(
        (tuple((*iper[1:-1], *reversed(jper[1:-1]))),),
    )
    if iper == jper:
        nlc = full
    else:
        acc = _check_nlc_one_open_line(tuple(jper[1:-1]), tuple(iper[1:-1]))
        nlc = full if acc != 0 else 0.0
    return (lc, nlc, full)


def _one_open_line_color_factors_exact(
    left: LCColorSector,
    right: LCColorSector,
    n_ord: int,
) -> tuple[Fraction, Fraction, Fraction]:
    iper = _coloured_word(left)
    jper = _coloured_word(right)
    lc = Fraction(NC) ** (n_ord - 1) if iper == jper else Fraction(0)
    full = _eval_trace_exact(
        (tuple((*iper[1:-1], *reversed(jper[1:-1]))),),
    )
    if iper == jper:
        nlc = full
    else:
        acc = _check_nlc_one_open_line(tuple(jper[1:-1]), tuple(iper[1:-1]))
        nlc = full if acc != 0 else Fraction(0)
    return (lc, nlc, full)


def _two_open_line_color_factors(
    color_plan: GenericColorPlan,
    left: LCColorSector,
    right: LCColorSector,
    n_ord: int,
) -> tuple[float, float, float]:
    reference_start = _two_line_reference_start(color_plan)
    iper = _rotate_to_reference_start(_coloured_word(left), reference_start)
    jper = _rotate_to_reference_start(_coloured_word(right), reference_start)
    reference = (
        _rotate_to_reference_start(
            _coloured_word(color_plan.sectors[0]), reference_start
        )
        if color_plan.sectors
        else iper
    )
    gi, ui = _two_line_gi_ui(color_plan, iper, reference)
    gj, uj = _two_line_gi_ui(color_plan, jper, reference)
    repeated_fundamental_species = _has_repeated_fundamental_species(color_plan)
    lc = 0.0
    if iper == jper:
        if ui == 1 and uj == 1:
            lc = float(NC ** (n_ord - 2))
        elif ui == 2 and uj == 2 and not repeated_fundamental_species:
            lc = float(NC ** (n_ord - 4) * 9.0)
        elif ui == 2 and uj == 2 and repeated_fundamental_species:
            lc = float(NC ** (n_ord - 2))
    full = _two_open_line_full_factor(
        iper,
        jper,
        n_ord=n_ord,
        gi=gi,
        gj=gj,
        ui=ui,
        uj=uj,
    )
    nlc = 0.0
    if abs(full) > 0.0:
        iper_adj, jper_adj = _two_line_ordered_adjoint_strings(
            color_plan,
            iper,
            jper,
        )
        iper_ord, jper_ord = _convert_two_line_adjoint_strings(
            n_ord,
            iper_adj,
            jper_adj,
        )
        acc = _check_nlc_two_open_lines_same_species(
            n_ord,
            iper_ord,
            jper_ord,
            gi,
            gj,
            ui,
            uj,
        )
        if acc != 0:
            nlc = full
    return (lc, nlc, full)


def _two_open_line_color_factors_exact(
    color_plan: GenericColorPlan,
    left: LCColorSector,
    right: LCColorSector,
    n_ord: int,
) -> tuple[Fraction, Fraction, Fraction]:
    reference_start = _two_line_reference_start(color_plan)
    iper = _rotate_to_reference_start(_coloured_word(left), reference_start)
    jper = _rotate_to_reference_start(_coloured_word(right), reference_start)
    reference = (
        _rotate_to_reference_start(
            _coloured_word(color_plan.sectors[0]), reference_start
        )
        if color_plan.sectors
        else iper
    )
    gi, ui = _two_line_gi_ui(color_plan, iper, reference)
    gj, uj = _two_line_gi_ui(color_plan, jper, reference)
    repeated_fundamental_species = _has_repeated_fundamental_species(color_plan)
    lc = Fraction(0)
    if iper == jper:
        if ui == 1 and uj == 1:
            lc = Fraction(NC) ** (n_ord - 2)
        elif ui == 2 and uj == 2 and not repeated_fundamental_species:
            lc = Fraction(9) * Fraction(NC) ** (n_ord - 4)
        elif ui == 2 and uj == 2 and repeated_fundamental_species:
            lc = Fraction(NC) ** (n_ord - 2)
    full = _two_open_line_full_factor_exact(
        iper,
        jper,
        n_ord=n_ord,
        gi=gi,
        gj=gj,
        ui=ui,
        uj=uj,
    )
    nlc = Fraction(0)
    if full:
        iper_adj, jper_adj = _two_line_ordered_adjoint_strings(
            color_plan,
            iper,
            jper,
        )
        iper_ord, jper_ord = _convert_two_line_adjoint_strings(
            n_ord,
            iper_adj,
            jper_adj,
        )
        acc = _check_nlc_two_open_lines_same_species(
            n_ord,
            iper_ord,
            jper_ord,
            gi,
            gj,
            ui,
            uj,
        )
        if acc != 0:
            nlc = full
    return (lc, nlc, full)


def _two_open_line_full_factor(
    iper: tuple[int, ...],
    jper: tuple[int, ...],
    *,
    n_ord: int,
    gi: int,
    gj: int,
    ui: int,
    uj: int,
) -> float:
    if ui == uj:
        traces = (
            tuple((*iper[1 : 1 + gi], *reversed(jper[1 : 1 + gj]))),
            tuple((*iper[gi + 3 : n_ord - 1], *reversed(jper[gj + 3 : n_ord - 1]))),
        )
        coeff = Fraction(1)
    elif (ui, uj) in {(1, 2), (2, 1)}:
        traces = (
            tuple(
                (
                    *iper[1 : 1 + gi],
                    *reversed(jper[gj + 3 : n_ord - 1]),
                    *iper[gi + 3 : n_ord - 1],
                    *reversed(jper[1 : 1 + gj]),
                )
            ),
        )
        coeff = Fraction(-1)
    else:
        return 0.0
    return _eval_trace(traces, coeff=coeff)


def _two_open_line_full_factor_exact(
    iper: tuple[int, ...],
    jper: tuple[int, ...],
    *,
    n_ord: int,
    gi: int,
    gj: int,
    ui: int,
    uj: int,
) -> Fraction:
    if ui == uj:
        traces = (
            tuple((*iper[1 : 1 + gi], *reversed(jper[1 : 1 + gj]))),
            tuple((*iper[gi + 3 : n_ord - 1], *reversed(jper[gj + 3 : n_ord - 1]))),
        )
        coeff = Fraction(1)
    elif (ui, uj) in {(1, 2), (2, 1)}:
        traces = (
            tuple(
                (
                    *iper[1 : 1 + gi],
                    *reversed(jper[gj + 3 : n_ord - 1]),
                    *iper[gi + 3 : n_ord - 1],
                    *reversed(jper[1 : 1 + gj]),
                )
            ),
        )
        coeff = Fraction(-1)
    else:
        return Fraction(0)
    return _eval_trace_exact(traces, coeff=coeff)


def _is_open_line_pair(
    left: LCColorSector,
    right: LCColorSector,
) -> bool:
    if left.kind != "open-lines" or right.kind != "open-lines":
        return False
    return len(left.open_color_lines) == len(right.open_color_lines)


def _multi_open_line_color_factors(
    color_plan: GenericColorPlan,
    left: LCColorSector,
    right: LCColorSector,
) -> tuple[float, float, float]:
    """Generic overlap of multi-open-line colour-flow tensors.

    Each sector is a product of open strings
    ``(T...T)_{i,\bar j}``.  The right sector is conjugated, so its adjoint
    string is traversed in reverse order, and the fundamental colour indices
    form closed alternating left/right cycles.  The resulting traces are reduced
    with the same Fierz machinery used for pure traces, but we keep powers of
    ``Nc`` symbolic so that NLC can be obtained by dropping terms beyond the
    first ``1/Nc**2`` suppression.
    """

    terms = _open_line_nc_power_terms(
        left.open_color_lines,
        right.open_color_lines,
    )
    if not terms:
        return (0.0, 0.0, 0.0)
    leading_power = color_plan.process.color_endpoints.pair_count + len(
        color_plan.process.adjoint_labels
    )
    lc = _eval_nc_terms(terms, min_power=leading_power)
    full = _eval_nc_terms(terms)
    # NLC selects matrix entries by their leading Nc power, but retains the
    # exact coefficient of every selected entry.  Truncating the coefficient
    # itself changes off-diagonal multi-line contractions once adjoints are
    # present (for example 8 -> 9 for three open lines plus one adjoint).
    nlc = full if max(terms) >= leading_power - 2 else 0.0
    return (lc, nlc, full)


def _multi_open_line_color_factors_exact(
    color_plan: GenericColorPlan,
    left: LCColorSector,
    right: LCColorSector,
) -> tuple[Fraction, Fraction, Fraction]:
    terms = _open_line_nc_power_terms(
        left.open_color_lines,
        right.open_color_lines,
    )
    if not terms:
        return (Fraction(0), Fraction(0), Fraction(0))
    leading_power = color_plan.process.color_endpoints.pair_count + len(
        color_plan.process.adjoint_labels
    )
    lc = _eval_nc_terms_exact(terms, min_power=leading_power)
    full = _eval_nc_terms_exact(terms)
    nlc = full if max(terms) >= leading_power - 2 else Fraction(0)
    return (lc, nlc, full)


def exact_color_contraction_factors(
    color_plan: GenericColorPlan,
    left: LCColorSector,
    right: LCColorSector,
    *,
    full_col_acc: int = 20,
) -> tuple[Fraction, Fraction, Fraction]:
    """Return exact reference-normalized ``(LC, NLC, full)`` color factors."""

    if color_plan.basis == "adjoint":
        return tuple(
            exact_color_contraction_factor(
                color_plan, left, right, accuracy=accuracy, full_col_acc=full_col_acc
            )
            for accuracy in ("lc", "nlc", "full")
        )
    open_line_count = color_plan.process.color_endpoints.pair_count
    n_ord = len(_coloured_word(left))
    if len(_coloured_word(right)) != n_ord:
        return (Fraction(0), Fraction(0), Fraction(0))
    if open_line_count == 0:
        return _pure_adjoint_color_factors_exact(
            left,
            right,
            n_ord,
            full_col_acc,
        )
    if open_line_count == 1:
        return _one_open_line_color_factors_exact(left, right, n_ord)
    if open_line_count == 2:
        return _two_open_line_color_factors_exact(color_plan, left, right, n_ord)
    if open_line_count >= 3 and _is_open_line_pair(left, right):
        return _multi_open_line_color_factors_exact(color_plan, left, right)
    return (Fraction(0), Fraction(0), Fraction(0))


def exact_color_contraction_factor(
    color_plan: GenericColorPlan,
    left: LCColorSector,
    right: LCColorSector,
    *,
    accuracy: str,
    full_col_acc: int = 20,
) -> Fraction:
    """Return one exact reference-normalized color factor."""

    if accuracy not in {"lc", "nlc", "full"}:
        return Fraction(0)
    if color_plan.basis == "adjoint":
        from .adjoint_kernel import adjoint_color_factor

        return Fraction(
            adjoint_color_factor(
                _coloured_word(left),
                _coloured_word(right),
                accuracy=accuracy,
                nc=NC,
                full_col_acc=full_col_acc,
            )
        )
    if color_plan.process.color_endpoints.pair_count == 0:
        n_ord = len(_coloured_word(left))
        if len(_coloured_word(right)) != n_ord:
            return Fraction(0)
        # LC/NLC have direct formulae and must not evaluate the full trace.
        return _pure_adjoint_color_factor_exact(
            left,
            right,
            n_ord,
            accuracy=accuracy,
            full_col_acc=full_col_acc,
        )
    values = exact_color_contraction_factors(
        color_plan,
        left,
        right,
        full_col_acc=full_col_acc,
    )
    if accuracy == "lc":
        return values[0]
    if accuracy == "nlc":
        return values[1]
    if accuracy == "full":
        return values[2]
    return Fraction(0)


def _open_line_nc_power_terms(
    left_lines: tuple[LCOpenColorLine, ...],
    right_lines: tuple[LCOpenColorLine, ...],
) -> Mapping[int, Fraction]:
    left_by_fundamental = {line.fundamental_label: line for line in left_lines}
    right_by_antifundamental = {
        line.antifundamental_label: line for line in right_lines
    }
    if set(left_by_fundamental) != {line.fundamental_label for line in right_lines}:
        return {}
    if {line.antifundamental_label for line in left_lines} != set(
        right_by_antifundamental
    ):
        return {}

    visited: set[int] = set()
    traces: list[tuple[int, ...]] = []
    for start in sorted(left_by_fundamental):
        if start in visited:
            continue
        current = start
        trace: list[int] = []
        while current not in visited:
            visited.add(current)
            left_line = left_by_fundamental.get(current)
            if left_line is None:
                return {}
            trace.extend(left_line.adjoint_labels)
            right_line = right_by_antifundamental.get(left_line.antifundamental_label)
            if right_line is None:
                return {}
            trace.extend(reversed(right_line.adjoint_labels))
            current = right_line.fundamental_label
        traces.append(tuple(trace))
    permutation_sign = -1 if (len(left_lines) - len(traces)) % 2 else 1
    return _simplify_trace_terms_nc_power(
        ((Fraction(permutation_sign), 0, tuple(traces)),)
    )


def _coloured_word(sector: LCColorSector) -> tuple[int, ...]:
    if sector.kind == "single-trace":
        return tuple(sector.trace_labels)
    if sector.kind == "open-lines":
        return tuple(sector.word_labels or sector.color_words[0])
    return ()


def _two_line_reference_start(color_plan: GenericColorPlan) -> int | None:
    if color_plan.process.fundamental_labels:
        return int(color_plan.process.fundamental_labels[0])
    return None


def _rotate_to_reference_start(
    word: tuple[int, ...],
    reference_start: int | None,
) -> tuple[int, ...]:
    if reference_start is None or reference_start not in word:
        return word
    offset = word.index(reference_start)
    return word[offset:] + word[:offset]


def _two_line_gi_ui(
    color_plan: GenericColorPlan,
    word: tuple[int, ...],
    reference_word: tuple[int, ...],
) -> tuple[int, int]:
    fundamental_endpoints = set(color_plan.process.fundamental_labels) | set(
        color_plan.process.antifundamental_labels
    )
    gi = 0
    for position in range(1, max(len(word) - 1, 1)):
        if word[position] in fundamental_endpoints:
            gi = position - 1
            break
    ui = 1
    if len(word) >= 2 and len(reference_word) >= 2:
        same_ends = word[0] == reference_word[0] and word[-1] == reference_word[-1]
        opposite_ends = word[0] != reference_word[0] and word[-1] != reference_word[-1]
        ui = 1 if same_ends or opposite_ends else 2
    return gi, ui


def _two_line_ordered_adjoint_strings(
    color_plan: GenericColorPlan,
    iper: tuple[int, ...],
    jper: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    fundamental_endpoints = set(color_plan.process.fundamental_labels) | set(
        color_plan.process.antifundamental_labels
    )
    return (
        tuple(label for label in iper if label not in fundamental_endpoints),
        tuple(label for label in jper if label not in fundamental_endpoints),
    )


def _convert_two_line_adjoint_strings(
    n_ord: int,
    iper: tuple[int, ...],
    jper: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Mirror AmpliCol's ordered-adjoint-string conversion.

    The NLC topology tests depend only on the relative order of adjoints.  The
    Fortran routine maps the largest external label to 1, the next largest to
    2, and so on, separately for the row and column strings.
    """

    expected = max(n_ord - 4, 0)
    if len(iper) != expected or len(jper) != expected:
        return iper, jper

    def convert(word: tuple[int, ...]) -> tuple[int, ...]:
        ordered = sorted(word, reverse=True)
        rank = {label: index + 1 for index, label in enumerate(ordered)}
        return tuple(rank[label] for label in word)

    return convert(iper), convert(jper)


def _check_nlc_two_open_lines_same_species(
    n_ord: int,
    iper: tuple[int, ...],
    jper: tuple[int, ...],
    ri: int,
    rj: int,
    ii: int,
    jj: int,
) -> int:
    """Port of AmpliCol's same-species two-open-line NLC topology filter.

    It returns 99 for LC-like entries, +/-1 for first subleading entries, and
    0 for NNLC-or-beyond entries.  The final NLC coefficient is still the full
    trace overlap; this function only decides whether that overlap belongs in
    the NLC-expanded matrix.
    """

    m = max(n_ord - 4, 0)
    if len(iper) != m or len(jper) != m:
        return 0
    if (ii, jj) in {(1, 2), (2, 1)}:
        temp = (
            iper[:ri]
            + tuple(reversed(jper[rj:]))
            + iper[ri:]
            + tuple(reversed(jper[:rj]))
        )
        return -1 if _nlc_pairing_is_planar(temp, m) else 0
    if (ii, jj) not in {(1, 1), (2, 2)}:
        return 0
    if ri == rj and iper == jper:
        return 99

    aa = iper[:ri]
    bb = iper[ri:]
    cc = jper[:rj]
    dd = jper[rj:]
    temp2 = aa + tuple(reversed(cc))
    temp3 = bb + tuple(reversed(dd))
    disjoint = not (set(temp2) & set(temp3))
    if disjoint:
        if aa and (bb == dd or not bb):
            return _check_nlc_subword(cc, aa, threshold=m - 4)
        if bb and (aa == cc or not aa):
            return _check_nlc_subword(dd, bb, threshold=m - rj - 4)
        return 0
    return _check_nlc_two_open_lines_overlap(aa, bb, cc, dd, m, ri, rj)


def _nlc_pairing_is_planar(word: tuple[int, ...], m: int) -> bool:
    if m == 0:
        return True
    positions: dict[int, list[int]] = {label: [] for label in range(1, m + 1)}
    for offset, label in enumerate(word):
        if label in positions:
            positions[label].append(offset)
    if any(len(item) != 2 for item in positions.values()):
        return False
    for first, pair in positions.items():
        if abs(pair[0] - pair[1]) % 2 != 1:
            return False
        for second in range(first + 1, m + 1):
            if not _intervals_disjoint_or_nested(tuple(pair), tuple(positions[second])):
                return False
    return True


def _intervals_disjoint_or_nested(
    first: tuple[int, int],
    second: tuple[int, int],
) -> bool:
    a1, a2 = first
    b1, b2 = second
    return (
        (a1 < b1 and a2 < b1)
        or (a1 > b2 and a2 > b2)
        or (a1 > b1 and a2 < b2)
        or (b1 > a1 and b2 < a2)
    )


def _check_nlc_subword(
    candidate: tuple[int, ...],
    target: tuple[int, ...],
    *,
    threshold: int,
) -> int:
    if candidate == target:
        return 99
    if not candidate or len(candidate) != len(target):
        return 0
    sign = _check_nlc(candidate, target)
    if sign == 0:
        return 0
    if sign < 0:
        return sign
    return sign


def _check_nlc_two_open_lines_overlap(
    aa: tuple[int, ...],
    bb: tuple[int, ...],
    cc: tuple[int, ...],
    dd: tuple[int, ...],
    m: int,
    ri: int,
    rj: int,
) -> int:
    """Remaining overlapping-sets branch of the two-open-line NLC check."""

    temp2 = aa + tuple(reversed(cc))
    temp3 = bb + tuple(reversed(dd))
    common = set(temp2) & set(temp3)
    if not common:
        return 0
    skipped = None
    ind_i = ind_j = -1
    for i, label in enumerate(temp2):
        if label not in common:
            continue
        ind_i = i
        ind_j = temp3.index(label)
        skipped = label
        break
    if skipped is None:
        return 0
    if len(temp2) == 1 or len(temp3) == 1:
        return 0

    perm: tuple[int, ...]
    if ind_i < ri and ind_j >= len(bb):
        # Common generator in the A-D pair.
        perm = (
            temp2[ind_i + 1 : ri]
            + temp2[ri : ri + rj]
            + temp2[:ind_i]
            + temp3[ind_j + 1 :]
            + temp3[: len(bb)]
            + temp3[len(bb) : ind_j]
        )
        itemp4 = temp2[:ind_i] + temp2[ind_i + 1 : ri]
        itemp5 = tuple(reversed(temp3[ind_j + 1 :])) + tuple(
            reversed(temp3[len(bb) : ind_j])
        )
        if rj == ri - 1 and itemp4 == cc:
            return 0
        if rj + 1 == ri and itemp5 == bb:
            return 0
    elif ind_i >= ri and ind_j < len(bb):
        # Common generator in the B-C pair.
        perm = (
            temp2[ind_i + 1 :]
            + temp2[:ri]
            + temp2[ri:ind_i]
            + temp3[ind_j + 1 : len(bb)]
            + temp3[len(bb) :]
            + temp3[:ind_j]
        )
        itemp6 = tuple(reversed(temp2[ind_i + 1 :])) + tuple(reversed(temp2[ri:ind_i]))
        itemp7 = temp3[:ind_j] + temp3[ind_j + 1 : len(bb)]
        if rj - 1 == ri and itemp6 == aa:
            return 0
        if rj == ri + 1 and itemp7 == dd:
            return 0
    else:
        return 0

    positions: dict[int, list[int]] = {label: [] for label in range(1, m + 1)}
    for offset, label in enumerate(perm):
        if label == skipped:
            continue
        if label in positions:
            positions[label].append(offset)
    for label, pair in positions.items():
        if label == skipped:
            continue
        if len(pair) != 2 or abs(pair[0] - pair[1]) % 2 != 1:
            return 0
    for first in range(1, m + 1):
        if first == skipped:
            continue
        for second in range(first + 1, m + 1):
            if second == skipped:
                continue
            if not _intervals_disjoint_or_nested(
                tuple(positions[first]),
                tuple(positions[second]),
            ):
                return 0
    return 1


def _has_repeated_fundamental_species(color_plan: GenericColorPlan) -> bool:
    pdg_by_label: dict[int, int] = {
        leg.label: abs(int(leg.outgoing_pdg))
        for leg in color_plan.process.legs
        if leg.outgoing_pdg is not None
    }
    fundamental_flavours = [
        pdg_by_label[label]
        for label in color_plan.process.fundamental_labels
        if label in pdg_by_label
    ]
    return len(set(fundamental_flavours)) < len(fundamental_flavours)


def _common_helicity_weight(
    left: ColorGroupDescriptor,
    right: ColorGroupDescriptor,
) -> float:
    if abs(left.helicity_weight - right.helicity_weight) > 1.0e-12:
        raise ValueError(
            "inconsistent helicity weights in color contraction group "
            f"{left.helicity_key!r}: group {left.group_id} has "
            f"{left.helicity_weight!r}, group {right.group_id} has "
            f"{right.helicity_weight!r}"
        )
    return left.helicity_weight
